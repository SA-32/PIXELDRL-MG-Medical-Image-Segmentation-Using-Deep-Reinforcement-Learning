"""
model.py
--------
PyTorch implementation of the network architecture described in:

  "Pixel level deep reinforcement learning for accurate and robust
   medical image segmentation" (PixelDRL-MG), Scientific Reports 2025.

Components (Fig. 1 of the paper):
  1. Feature Extraction Module   -> VGG16 with half the channel count,
                                     dilation convolutions, first 23 layers
                                     (conv1-1 .. conv4-3), multi-scale
                                     concatenation of conv1-2/conv2-2/conv3-3.
  2. Self-Attention Module (SAM) -> non-local style attention (Eq. 1).
  3. Policy Network / Value Network -> both built with dilated convolutions
                                     (DC) in every layer, output resolution
                                     == input resolution.
  4. PA3C wrapper (Pixel-level Asynchronous Advantage Actor-Critic) that
     ties everything together and exposes a `step()` that returns
     policy logits pi(a|s), value V(s), and the sampled/greedy action.

Action space |A| = 2:
    action 0 -> set pixel to background (0)
    action 1 -> do nothing (keep current foreground value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- #
# 1. Feature extraction module: VGG16 (half channels) + dilation convolutions
# --------------------------------------------------------------------------- #
class VGG16HalfDilated(nn.Module):
    """
    VGG16 front-end (conv1-1 .. conv4-3, i.e. the first 23 layers of
    torchvision's vgg16.features) with:
      * half the number of channels at every conv layer,
      * 3x3 dilated convolutions instead of plain convolutions,
      * standard max-pooling kept (so that the paper's stated 1/2, 1/4,
        1/8, 1/16 multi-scale feature maps are still produced).

    Multi-scale source layers returned: conv1-2, conv2-2, conv3-3
    (as specified in the paper), each later upsampled to a common
    resolution and concatenated by the policy/value networks.
    """

    def __init__(self, in_channels=1, dilation=2):
        super().__init__()

        def dconv(c_in, c_out, d=dilation):
            # 3x3 dilated conv, padding chosen to keep spatial size fixed
            return nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d)

        # Half of the standard VGG16 channel counts: 64,64 / 128,128 / 256,256,256 / 512,512,512
        c = [32, 64, 128, 256]  # half of 64,128,256,512

        # Block 1 -> conv1-1, conv1-2
        self.conv1_1 = dconv(in_channels, c[0])
        self.conv1_2 = dconv(c[0], c[0])
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 1/2

        # Block 2 -> conv2-1, conv2-2
        self.conv2_1 = dconv(c[0], c[1])
        self.conv2_2 = dconv(c[1], c[1])
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 1/4

        # Block 3 -> conv3-1, conv3-2, conv3-3
        self.conv3_1 = dconv(c[1], c[2])
        self.conv3_2 = dconv(c[2], c[2])
        self.conv3_3 = dconv(c[2], c[2])
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 1/8

        # Block 4 -> conv4-1, conv4-2, conv4-3
        self.conv4_1 = dconv(c[2], c[3])
        self.conv4_2 = dconv(c[3], c[3])
        self.conv4_3 = dconv(c[3], c[3])
        self.pool4 = nn.MaxPool2d(2, 2)  # -> 1/16

        self.relu = nn.ReLU(inplace=True)
        self.out_channels = c[0] + c[1] + c[2]  # conv1-2 + conv2-2 + conv3-3

    def forward(self, x):
        h = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(h))          # 1/1 (pre-pool) scale
        h = self.pool1(conv1_2)                        # 1/2

        h = self.relu(self.conv2_1(h))
        conv2_2 = self.relu(self.conv2_2(h))           # 1/2 scale
        h = self.pool2(conv2_2)                        # 1/4

        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        conv3_3 = self.relu(self.conv3_3(h))           # 1/4 scale
        h = self.pool3(conv3_3)                        # 1/8

        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        conv4_3 = self.relu(self.conv4_3(h))           # 1/8 scale
        h = self.pool4(conv4_3)                        # 1/16 (unused directly,
                                                         # kept for fidelity to
                                                         # "1/2,1/4,1/8,1/16" text)

        target_size = conv1_2.shape[-2:]
        f2 = F.interpolate(conv2_2, size=target_size, mode="bilinear", align_corners=False)
        f3 = F.interpolate(conv3_3, size=target_size, mode="bilinear", align_corners=False)

        # Multi-scale concatenation (conv1-2, conv2-2, conv3-3), as stated in the paper
        fatt = torch.cat([conv1_2, f2, f3], dim=1)
        return fatt


# --------------------------------------------------------------------------- #
# 2. Self-Attention Module (SAM), Eq. (1): Matt = Softmax(W*conv(fatt)+fatt)
#    Implemented as a non-local self-attention block (query/key/value),
#    matching Fig. 1's diagram (1x1 convs -> reshape -> matmul -> softmax
#    -> matmul -> 1x1 conv -> residual add).
# --------------------------------------------------------------------------- #
class SelfAttentionModule(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        inter = max(channels // reduction, 1)
        self.query_conv = nn.Conv2d(channels, inter, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, inter, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fatt):
        B, C, H, W = fatt.shape
        q = self.query_conv(fatt).view(B, -1, H * W).permute(0, 2, 1)   # (B, HW, C/2)
        k = self.key_conv(fatt).view(B, -1, H * W)                     # (B, C/2, HW)
        v = self.value_conv(fatt).view(B, C, H * W)                    # (B, C, HW)

        attn = self.softmax(torch.bmm(q, k))                           # (B, HW, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)      # (B, C, H, W)
        out = self.out_conv(out)

        # Residual connection, matching Eq. (1)'s "+ fatt"
        return out + fatt


# --------------------------------------------------------------------------- #
# 3. Policy / Value networks: every layer uses dilated convolutions (DC),
#    output resolution == input image resolution.
# --------------------------------------------------------------------------- #
class DilatedConvHead(nn.Module):
    """
    Shared building block for the policy and value heads: a small stack of
    dilated convolutions (to gather neighbouring-pixel information, as
    described in the "Pixel-level asynchronous advantage actor-critic"
    section), followed by a 1x1 projection to `out_channels`, then
    upsampling back to the original image resolution.
    """

    def __init__(self, in_channels, hidden=128, out_channels=2, n_layers=4, dilation=2):
        super().__init__()
        layers = []
        c = in_channels
        for i in range(n_layers):
            layers += [
                nn.Conv2d(c, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True),
            ]
            c = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, s, out_size):
        h = self.body(s)
        h = self.head(h)
        h = F.interpolate(h, size=out_size, mode="bilinear", align_corners=False)
        return h


class PolicyNetwork(DilatedConvHead):
    """Outputs pi(a|s): 2 channels (background / do-nothing), softmax over channel dim."""

    def __init__(self, in_channels, hidden=128, n_layers=4, dilation=2, n_actions=2):
        super().__init__(in_channels, hidden=hidden, out_channels=n_actions,
                          n_layers=n_layers, dilation=dilation)

    def forward(self, s, out_size):
        logits = super().forward(s, out_size)          # (B, |A|, H, W)
        probs = F.softmax(logits, dim=1)
        return probs


class ValueNetwork(DilatedConvHead):
    """Outputs V(s): 1 channel, per-pixel state value."""

    def __init__(self, in_channels, hidden=128, n_layers=4, dilation=2):
        super().__init__(in_channels, hidden=hidden, out_channels=1,
                          n_layers=n_layers, dilation=dilation)

    def forward(self, s, out_size):
        v = super().forward(s, out_size)                # (B, 1, H, W)
        return v.squeeze(1)                              # (B, H, W)


# --------------------------------------------------------------------------- #
# 4. PixelDRL-MG: feature extractor + SAM + PA3C (policy & value networks)
# --------------------------------------------------------------------------- #
class PixelDRL_MG(nn.Module):
    """
    Full model, matching Fig. 1:

        X^(t) --[VGG16-half+DC]--> s'^(t) --[SAM]--> s^(t)
                                                 |--> Policy Network -> pi(a^(t)|s^(t))
                                                 |--> Value Network  -> V(s^(t))

    `use_sam` / `use_dc` flags are provided to reproduce the ablation study
    in Table 3 (PA3C, PA3C+SAM, PA3C+DC, PA3C+SAM+DC == full model).
    """

    def __init__(self, in_channels=1, n_actions=2, use_sam=True, use_dc=True,
                 policy_hidden=128, value_hidden=128, n_layers=4):
        super().__init__()
        self.use_sam = use_sam
        self.use_dc = use_dc

        self.feature_extractor = VGG16HalfDilated(in_channels=in_channels)
        feat_channels = self.feature_extractor.out_channels

        if use_sam:
            self.sam = SelfAttentionModule(feat_channels)
        else:
            self.sam = nn.Identity()

        dilation = 2 if use_dc else 1
        self.policy_net = PolicyNetwork(feat_channels, hidden=policy_hidden,
                                         n_layers=n_layers, dilation=dilation,
                                         n_actions=n_actions)
        self.value_net = ValueNetwork(feat_channels, hidden=value_hidden,
                                       n_layers=n_layers, dilation=dilation)

    def forward(self, x):
        """
        x: (B, C, H, W) current temporary input X^(t)
        returns: pi (B, |A|, H, W), V (B, H, W)
        """
        out_size = x.shape[-2:]
        s_prime = self.feature_extractor(x)     # s'^(t)
        s = self.sam(s_prime)                   # s^(t)
        pi = self.policy_net(s, out_size)       # pi(a^(t)|s^(t))
        v = self.value_net(s, out_size)         # V(s^(t))
        return pi, v

    @torch.no_grad()
    def act_greedy(self, x):
        """Greedy action selection (used at inference / evaluation time)."""
        pi, v = self.forward(x)
        action = torch.argmax(pi, dim=1)        # (B, H, W) in {0, 1}
        return action, pi, v

    def act_sample(self, x):
        """Stochastic action sampling (used during training, matches A3C)."""
        pi, v = self.forward(x)
        dist = torch.distributions.Categorical(probs=pi.permute(0, 2, 3, 1))
        action = dist.sample()                  # (B, H, W)
        log_prob = dist.log_prob(action)         # (B, H, W)
        entropy = dist.entropy()                 # (B, H, W)
        return action, log_prob, entropy, v
