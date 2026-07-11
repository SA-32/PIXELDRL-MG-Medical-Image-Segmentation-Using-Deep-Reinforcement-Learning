# PixelDRL-MG — PyTorch Implementation

A PyTorch implementation of:

> Liu, Y., Yuan, D., Xu, Z., Zhan, Y., Zhang, H., Lu, J., & Lukasiewicz, T.
> **"Pixel level deep reinforcement learning for accurate and robust medical
> image segmentation."** *Scientific Reports* 15, 8213 (2025).
> https://doi.org/10.1038/s41598-025-92117-2

This follows the paper's architecture and training algorithm (Fig. 1, Fig. 2,
Algorithm 1) as closely as is practical in a single, runnable codebase.

## Files

| File           | Paper section implemented |
|----------------|----------------------------|
| `model.py`     | Feature extraction module (VGG16, half channels, dilated convs), Self-Attention Module (Eq. 1), Policy/Value networks with dilated convolutions (PA3C) |
| `env.py`       | Dynamic iterative update policy, reward function (Eq. 12), return/advantage computation (Eqs. 8–16) |
| `train.py`     | Algorithm 1 (training loop) |
| `evaluate.py`  | Test-time evaluation (Table 2 protocol) |
| `metrics.py`   | DICE, PPV, SEN, IoU, BIoU, HD95 (as defined in the "Evaluation" subsection) |
| `dataset.py`   | 2D slice dataset loader, 70/10/20 split, k-shot support for Table 4's extreme-data experiments |
| `config.py`    | All hyperparameters from "Implementation details" |

## Architecture summary (Fig. 1)

```
X^(t) --> [VGG16-half, dilated convs, 23 layers] --> s'^(t)
      --> [Self-Attention Module]                --> s^(t)
      --> [Policy Network (dilated convs)]        --> pi(a^(t) | s^(t))   (2 actions)
      --> [Value Network  (dilated convs)]        --> V(s^(t))
      --> sample a^(t) ~ pi                       --> update mask m^(t+1)
      --> X^(t+1) = image * m^(t+1)
      --> repeat for t_max steps
```

Actions: `0` = set pixel to background, `1` = "do nothing" (keep the pixel's
current foreground/background state). This matches the paper's design of
directly generating pixel-by-pixel masks instead of thresholding a
probability map.

## Quick start

```bash
pip install torch torchvision scipy pillow

# Expected data layout:
#   my_dataset/images/*.png
#   my_dataset/masks/*.png     (same filenames, binary masks)

python train.py --train_root my_dataset --val_root my_dataset

# Extreme-data-constraint experiments (Table 4):
python train.py --train_root my_dataset --k_shot 50
python train.py --train_root my_dataset --k_shot 100

# Ablation study (Table 3): PA3C / PA3C+SAM / PA3C+DC / PA3C+SAM+DC
python train.py --train_root my_dataset --no_sam --no_dc   # PA3C only
python train.py --train_root my_dataset --no_dc            # PA3C + SAM
python train.py --train_root my_dataset --no_sam            # PA3C + DC
python train.py --train_root my_dataset                     # full model

# Evaluate a checkpoint (reproduces Table 2's metrics)
python evaluate.py --test_root my_dataset --ckpt checkpoints/pixeldrl_mg_epoch200.pt
```

A tiny synthetic "toy" dataset (random noise images with circular masks) is
included under `data/toy/` purely so you can sanity-check that the full
pipeline (`dataset.py` → `model.py` → `env.py` → `train.py` → `metrics.py`)
runs end-to-end without errors:

```bash
python train.py --train_root data/toy --val_root data/toy --epochs 2
```

## Where this implementation is faithful vs. where it necessarily approximates

The paper leaves several implementation-level details unspecified (exact
channel widths beyond "half of VGG16", exact dilation rates, exact
policy/value head depth, and the literal asynchronous multi-process training
of A3C). This code is faithful to everything that **is** specified, and
makes clearly-documented, standard choices for what isn't:

* **Architecture** — VGG16 first 23 layers (conv1-1..conv4-3) with half
  channels and 3×3 dilated convolutions; conv1-2/conv2-2/conv3-3 multi-scale
  concatenation; SAM implemented as the non-local self-attention block shown
  in Fig. 1 (1×1 convs → reshape → matmul → softmax → matmul → 1×1 conv →
  residual add), exactly matching Eq. (1)'s `Softmax(W·f + f)` form; policy
  and value networks are stacks of dilated convolutions producing
  full-resolution, per-pixel outputs (2 action channels for the policy,
  1 value channel for the value network), as specified.

* **RL formulation** — the dynamic iterative update policy, the reward
  function (Eq. 12), and the matrix-form return/advantage computation
  (Eqs. 8–16) are implemented directly. The paper notes that the
  neighbourhood weights `ω_{i-j}` in Eq. (8) "are essentially convolution
  filter weights, which can be learned concurrently with θ_p and θ_v" — this
  is implemented as a small learned depthwise convolution
  (`NeighborhoodAggregator` in `env.py`) applied to the next-step value map.

* **Asynchrony** — the original A3C/PA3C description launches multiple
  asynchronous worker threads, each with a local copy of the parameters,
  that periodically push gradients to shared global parameters. This
  implementation instead performs the mathematically equivalent
  **synchronous, batched** update: one rollout of `t_max` steps over a
  minibatch of images, followed by a single gradient step on the shared
  parameters. This is a standard, widely-used simplification of A3C (it is,
  in fact, exactly what "A2C" — the synchronous variant of A3C — does) and
  preserves every loss term and gradient computation in Algorithm 1; it
  simply removes the parallel-worker infrastructure, which contributes
  nothing to the model's mathematics.

* **HD95 / BIoU** — implemented from their standard definitions (Karimi &
  Salcudean 2019 for HD95; Cheng et al., CVPR 2021 for Boundary IoU), as
  cited by the paper.

## Reproducing the paper's exact numbers

To reproduce Table 2 / 3 / 4 numbers you will need the paper's actual
datasets (Cardiac MRI, Brain MRI/LGG, Hecktor), preprocessed as described in
"Datasets" (slice 3D volumes to 2D, drop empty slices, 70/10/20 split) using
`slice_volume_to_2d()` in `dataset.py`, and to train for the full 200 epochs
with `t_max=10`, `gamma=0.95`, Adam lr=1e-3 (×0.9 every 25 epochs), batch
size 2, as set by default in `config.py`.
