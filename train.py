"""
train.py
--------
Implements Algorithm 1 ("Training PixelDRL-MG Using a Dynamic Iterative
Update Policy") from the paper.

Mapping from the paper's Algorithm 1 to this code:

  Input: T_max, t_max, (X, G), segmentation network M(theta_s)          -> `train()` args
  Line 1: sample a training batch (x, g)                                -> dataloader batch
  Line 2/6: reset gradients                                             -> optimizer.zero_grad()
  Line 3/7: thread-specific params theta_p', theta_v', theta_s'         -> (single synchronous
             synchronized with global theta_p, theta_v, theta_s            copy of the model;
                                                                             true A3C's per-thread
                                                                             copies are replaced
                                                                             by a batched rollout,
                                                                             see module docstring
                                                                             in env.py)
  Line 9: obtain state s_i^(t) for all i                                 -> PixelEnv.init_mask /
                                                                             temp_input
  Lines 10-16: rollout for t_start..t_max, performing actions, storing   -> rollout loop below
               rewards/new states, computing segmentation metrics
  Line 17: bootstrap R_i from terminal or non-terminal value             -> compute_targets()
  Lines 18-26: backward accumulation of discounted returns & gradients   -> policy/value losses
               for theta_p, theta_v, theta_s                                + one backward() call
  Line 27-28: update parameters, update M via gradient descent           -> optimizer.step()
"""

import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import MedicalSegmentationDataset
from env import PixelEnv, NeighborhoodAggregator, compute_targets
from metrics import compute_all_metrics, average_metrics
from model import PixelDRL_MG


def rollout_episode(model, env, aggregator, image, gt, t_max, gamma, device):
    """
    Runs one t_max-step episode of the dynamic iterative update policy for
    an entire batch of images, collecting log-probs, entropies, values and
    rewards needed to compute the PA3C losses (Eqs. 8-16).

    Returns:
        log_probs: list[T] of (B, H, W)
        entropies: list[T] of (B, H, W)
        values:    list[T+1] of (B, H, W)   (values[0..T-1] on-policy, values[T] terminal)
        rewards:   list[T] of (B, H, W)
        final_mask: (B, 1, H, W) segmentation output f^(T)
    """
    B, _, H, W = image.shape
    mask = env.init_mask(image)  # m^(0), shape (B, 1, H, W)

    log_probs, entropies, values, rewards = [], [], [], []

    for t in range(t_max):
        x_t = env.temp_input(image, mask)               # X^(t)
        action, log_prob, entropy, value = model.act_sample(x_t)
        # action, log_prob, entropy, value: (B, H, W) / (B, H, W) / (B, H, W) / (B, H, W)

        new_mask_2d = env.step(mask.squeeze(1), action)  # m^(t+1), (B, H, W)
        r = env.reward(mask.squeeze(1), new_mask_2d, gt.squeeze(1))  # r^(t), (B, H, W)

        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        rewards.append(r)

        mask = new_mask_2d.unsqueeze(1)                  # (B, 1, H, W)

    # Terminal value estimate (Algorithm 1, line 17: V(s^(t); theta') for
    # non-terminal states, used to bootstrap the final step's return).
    with torch.no_grad():
        x_T = env.temp_input(image, mask)
        _, v_T = model(x_T)
    values.append(v_T)

    return log_probs, entropies, values, rewards, mask


def compute_losses(log_probs, entropies, values, returns, entropy_coef, value_loss_coef):
    """
    PA3C losses in matrix form (Eqs. 13-16):

        A(a,s) = R - V(s)                                    (Eq. 14)
        dtheta_v ~ (R - V(s))^2                               (Eq. 13/9)
        dtheta_p ~ -log(pi(a|s)) * A(a,s)                     (Eq. 15/11)
        dtheta_s = dtheta_p + dtheta_v (shared backbone)      (Eq. 16)
    """
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    T = len(returns)

    for t in range(T):
        R = returns[t].detach()
        V = values[t]
        advantage = R - V                                    # Eq. (14) / (10)

        value_loss = value_loss + (advantage ** 2).mean()     # Eq. (13) / (9)
        policy_loss = policy_loss + (-log_probs[t] * advantage.detach()).mean()  # Eq. (15)/(11)
        entropy_loss = entropy_loss - entropies[t].mean()

    policy_loss = policy_loss / T
    value_loss = value_loss / T
    entropy_loss = entropy_loss / T

    total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss.item(), value_loss.item()


def train(cfg: Config, train_root, val_root=None, k_shot=None):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    train_ds = MedicalSegmentationDataset(train_root, split="train",
                                           image_size=cfg.image_size,
                                           seed=cfg.seed, k_shot=k_shot)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)

    val_loader = None
    if val_root is not None:
        val_ds = MedicalSegmentationDataset(val_root, split="val",
                                             image_size=cfg.image_size, seed=cfg.seed)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = PixelDRL_MG(in_channels=cfg.in_channels, n_actions=cfg.n_actions,
                         use_sam=cfg.use_sam, use_dc=cfg.use_dc,
                         policy_hidden=cfg.policy_hidden, value_hidden=cfg.value_hidden,
                         n_layers=cfg.n_layers).to(device)

    aggregator = NeighborhoodAggregator().to(device)
    env = PixelEnv(device)

    params = list(model.parameters()) + list(aggregator.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_every,
                                                 gamma=cfg.lr_decay_gamma)

    global_step = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_start = time.time()
        running_loss, running_dice = 0.0, 0.0

        for batch_idx, (image, gt) in enumerate(train_loader):
            image, gt = image.to(device), gt.to(device)

            optimizer.zero_grad()

            log_probs, entropies, values, rewards, final_mask = rollout_episode(
                model, env, aggregator, image, gt, cfg.t_max, cfg.gamma, device
            )

            returns = compute_targets(rewards, values, cfg.gamma, aggregator)

            total_loss, p_loss, v_loss = compute_losses(
                log_probs, entropies, values, returns, cfg.entropy_coef, cfg.value_loss_coef
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                pred = (final_mask > 0.5).float()
                dice = 0.0
                for b in range(pred.shape[0]):
                    m = compute_all_metrics(pred[b, 0], gt[b, 0])
                    dice += m["DICE"]
                dice /= pred.shape[0]

            running_loss += total_loss.item()
            running_dice += dice
            global_step += 1

            if batch_idx % cfg.log_every == 0:
                print(f"[epoch {epoch:03d}][batch {batch_idx:04d}] "
                      f"loss={total_loss.item():.4f} (p={p_loss:.4f}, v={v_loss:.4f}) "
                      f"dice={dice:.4f}")

        scheduler.step()
        n_batches = len(train_loader)
        print(f"== Epoch {epoch:03d} done in {time.time()-epoch_start:.1f}s | "
              f"avg_loss={running_loss/n_batches:.4f} avg_dice={running_dice/n_batches:.4f} ==")

        if val_loader is not None and (epoch + 1) % 10 == 0:
            evaluate(model, env, val_loader, cfg, device, tag=f"val_epoch{epoch}")

        if (epoch + 1) % 25 == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"pixeldrl_mg_epoch{epoch+1}.pt")
            torch.save({"model": model.state_dict(),
                        "aggregator": aggregator.state_dict(),
                        "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    return model, aggregator


@torch.no_grad()
def evaluate(model, env, data_loader, cfg: Config, device, tag="test"):
    """
    Runs the greedy (non-stochastic) dynamic iterative update policy for
    t_max steps and reports DICE/PPV/SEN/IoU/BIoU/HD95, matching Table 2's
    evaluation protocol.
    """
    model.eval()
    all_metrics = []
    for image, gt in data_loader:
        image, gt = image.to(device), gt.to(device)
        mask = env.init_mask(image)
        for t in range(cfg.t_max):
            x_t = env.temp_input(image, mask)
            action, _, _ = model.act_greedy(x_t)
            new_mask_2d = env.step(mask.squeeze(1), action)
            mask = new_mask_2d.unsqueeze(1)

        pred = (mask > 0.5).float()
        for b in range(pred.shape[0]):
            all_metrics.append(compute_all_metrics(pred[b, 0], gt[b, 0]))

    avg = average_metrics(all_metrics)
    print(f"[{tag}] " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    model.train()
    return avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True,
                         help="Path to dataset root containing images/ and masks/ subfolders")
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--k_shot", type=int, default=None,
                         help="For the extreme-data-constraint experiments (Table 4): 50 or 100")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_sam", action="store_true")
    parser.add_argument("--no_dc", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.no_sam:
        cfg.use_sam = False
    if args.no_dc:
        cfg.use_dc = False

    train(cfg, args.train_root, args.val_root, k_shot=args.k_shot)
