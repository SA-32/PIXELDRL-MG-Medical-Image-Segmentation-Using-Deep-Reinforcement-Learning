import torch
import torch.nn as nn


class NeighborhoodAggregator(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, v_next):
        # Accept either (B,H,W) or (B,1,H,W)
        if v_next.ndim == 3:
            v_next = v_next.unsqueeze(1)
        elif v_next.ndim != 4:
            raise ValueError(f"Unexpected shape: {v_next.shape}")

        return self.conv(v_next).squeeze(1)


class PixelEnv():
    
    def __init__(self, device):
        self.device = device

    @staticmethod
    def init_mask(image):
        """m^(0) = ones_like(image); s_i^(0) = I_i as stated in the paper."""
        return torch.ones_like(image)

    @staticmethod
    def temp_input(image, mask):
        return image * mask

    @staticmethod
    def step(mask, action):
        return action.float()

    @staticmethod
    def reward(prev_mask, new_mask, gt):
        prev_err = (prev_mask - gt) ** 2
        new_err = (new_mask - gt) ** 2
        return prev_err - new_err       # (B, H, W)


def compute_targets(rewards, values, gamma, aggregator: NeighborhoodAggregator):
    T = len(rewards)
    returns = [None] * T
    # Bootstrap from the last value estimate (Algorithm 1, line 17-19)
    R = values[T].detach()
    for t in reversed(range(T)):
        # Eq. (8): R_i^(t) = r_i^(t) + gamma * sum_{j in N(i)} w_{i-j} V(s_j^{(t+1)})
        # print(values[t+1].shape)
        bootstrap = aggregator(values[t + 1].detach()) if t + 1 <= T else R
        R = rewards[t] + gamma * bootstrap
        returns[t] = R
    return returns
