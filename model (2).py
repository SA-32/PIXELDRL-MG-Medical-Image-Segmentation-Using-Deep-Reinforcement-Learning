import torch
import torch.nn as nn
import torch.nn.functional as F


def c(ch: int, scale: float) -> int:

    return max(8, int(round(ch * scale)))


def conv_bn_relu(in_ch, out_ch, kernel_size=3, dilation=1, stride=1):
    
    padding = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------- #
#  Feature extraction module: VGG16 with half channels + dilated convolutions
# --------------------------------------------------------------------------- #

class VGG16HalfEncoder(nn.Module):
    
    def __init__(self, in_channels: int = 1, scale: float = 0.5, dilation: int = 1):
        super().__init__()
        c1, c2, c3, c4 = c(64, scale), c(128, scale), c(256, scale), c(512, scale)

        # Block 1 (full resolution) -> conv1-2 is the first skip connection
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, c1, dilation=dilation),
            conv_bn_relu(c1, c1, dilation=dilation),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 1/2

        # Block 2 (1/2 resolution) -> conv2-2 is the second skip connection
        self.block2 = nn.Sequential(
            conv_bn_relu(c1, c2, dilation=dilation),
            conv_bn_relu(c2, c2, dilation=dilation),
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 1/4

        # Block 3 (1/4 resolution) -> conv3-3 is the third skip connection
        self.block3 = nn.Sequential(
            conv_bn_relu(c2, c3, dilation=dilation),
            conv_bn_relu(c3, c3, dilation=dilation),
            conv_bn_relu(c3, c3, dilation=dilation),
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 1/8

        # Block 4 (1/8 resolution) -> conv4-3, deepest feature fed to SAM
        self.block4 = nn.Sequential(
            conv_bn_relu(c3, c4, dilation=dilation),
            conv_bn_relu(c4, c4, dilation=dilation),
            conv_bn_relu(c4, c4, dilation=dilation),
        )
        self.pool4 = nn.MaxPool2d(2, 2)  # -> 1/16

        self.out_channels = dict(c1=c1, c2=c2, c3=c3, c4=c4)

    def forward(self, x):
        f1 = self.block1(x)        # conv1-2, full res      -> skip
        p1 = self.pool1(f1)
        f2 = self.block2(p1)       # conv2-2, 1/2 res        -> skip
        p2 = self.pool2(f2)
        f3 = self.block3(p2)       # conv3-3, 1/4 res        -> skip
        p3 = self.pool3(f3)
        f4 = self.block4(p3)       # conv4-3, 1/8 res        -> fed to SAM
        p4 = self.pool4(f4)
        return p1, p2, p3, p4


# --------------------------------------------------------------------------- #
#  Self-Attention Module (SAM) -- non-local self-attention, Eq. (1) + Fig. 1
# --------------------------------------------------------------------------- #

class SelfAttentionModule(nn.Module):

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        inter = max(1, channels // reduction)
        self.query = nn.Conv2d(channels, inter, kernel_size=1)
        self.key = nn.Conv2d(channels, inter, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_att):
        b, ch, h, w = f_att.shape
        q = self.query(f_att).view(b, -1, h * w).permute(0, 2, 1)      # (B, HW, C/2)
        k = self.key(f_att).view(b, -1, h * w)                          # (B, C/2, HW)
        v = self.value(f_att).view(b, ch, h * w).permute(0, 2, 1)       # (B, HW, C)

        attn = self.softmax(torch.bmm(q, k))                            # (B, HW, HW)
        out = torch.bmm(attn, v)                                        # (B, HW, C)
        out = out.permute(0, 2, 1).contiguous().view(b, ch, h, w)
        out = self.out_proj(out)
        return out + f_att   # residual connection ("+ f_att" in Eq. 1 / Fig. 1)


# --------------------------------------------------------------------------- #
#  Shared trunk: encoder + SAM  (parameters theta_s)
# --------------------------------------------------------------------------- #

class SharedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG16HalfEncoder(in_channels = 1, scale = 0.5, dilation = 2)
        self.sam = SelfAttentionModule(256, reduction = 2)
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        s = self.sam(f4)          # s^(t): global-information-enriched bottleneck feature
        return f1, f2, f3, s


# --------------------------------------------------------------------------- #
#  Decoder head shared structure: dilated-conv upsampling decoder with skips
# --------------------------------------------------------------------------- #

class DilatedDecoderHead(nn.Module):
    
    def __init__(self, enc_channels: dict, out_channels: int, dilation: int = 2):
        super().__init__()
        c1, c2, c3, c4 = (enc_channels["c1"], enc_channels["c2"],
                          enc_channels["c3"], enc_channels["c4"])

        self.up4to3 = conv_bn_relu(c4, c3, dilation=dilation)
        self.dec3   = conv_bn_relu(c3 + c3, c3, dilation=dilation)

        self.up3to2 = conv_bn_relu(c3, c2, dilation=dilation)
        self.dec2   = conv_bn_relu(c2 + c2, c2, dilation=dilation)

        self.up2to1 = conv_bn_relu(c2, c1, dilation=dilation)
        self.dec1   = conv_bn_relu(c1 + c1, c1, dilation=dilation)

        self.head   = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, f1, f2, f3, s):
        x = F.interpolate(s, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4to3(x)
        x = self.dec3(torch.cat([x, f3], dim=1))

        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3to2(x)
        x = self.dec2(torch.cat([x, f2], dim=1))

        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2to1(x)
        x = self.dec1(torch.cat([x, f1], dim=1))

        x =  F.interpolate(x, size= 64, mode="bilinear", align_corners=False)

        return self.head(x)   # resolution == input resolution


class PolicyHead(nn.Module):
  
    def __init__(self, enc_channels, num_actions=2, dilation=2):
        super().__init__()
        self.decoder = DilatedDecoderHead(enc_channels, num_actions, dilation)

    def forward(self, f1, f2, f3, s):
        logits = self.decoder(f1, f2, f3, s)          # (B, |A|, H, W)
        probs = F.softmax(logits, dim=1)
        return probs, logits


class ValueHead(nn.Module):

    def __init__(self, enc_channels, dilation=2):
        super().__init__()
        self.decoder = DilatedDecoderHead(enc_channels, 1, dilation)

    def forward(self, f1, f2, f3, s):
        return self.decoder(f1, f2, f3, s)             # (B, 1, H, W)


# ----------------------------------------------------------------------------------- #
#  Full PA3C model = SharedTrunk (theta_s) + PolicyHead (theta_p) + ValueHead (theta_v)
# ----------------------------------------------------------------------------------- #

class PA3C(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.trunk = SharedTrunk()
        ch = self.trunk.out_channels
        self.policy_head = PolicyHead(ch, num_actions = 2, dilation = 2)
        self.value_head = ValueHead(ch, dilation = 2)

    def forward(self, x):
        f1, f2, f3, s = self.trunk(x)
        probs, logits = self.policy_head(f1, f2, f3, s)
        value = self.value_head(f1, f2, f3, s)
        return probs, logits, value

    # convenience accessors matching the paper's theta_p / theta_v / theta_s split
    def theta_s(self):
        return self.trunk.parameters()

    def theta_p(self):
        return self.policy_head.parameters()

    def theta_v(self):
        return self.value_head.parameters()


class PixelDRLMG(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pa3c = PA3C()

    def forward(self, x):
        return self.pa3c(x)

    def act(self, x, greedy: bool = False):
        
        probs, logits, value = self.pa3c(x)
        dist = torch.distributions.Categorical(probs=probs.permute(0, 2, 3, 1))
        if greedy:
            action = probs.argmax(dim=1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)          # (B, H, W)
        entropy = dist.entropy()                  # (B, H, W)
        return action, log_prob, entropy, value.squeeze(1)
