"""
evaluate.py
-----------
Standalone script to load a trained PixelDRL-MG checkpoint and evaluate it
on a held-out test split, reproducing the metrics reported in Table 2.

Usage:
    python evaluate.py --test_root /path/to/dataset --ckpt checkpoints/pixeldrl_mg_epoch200.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MedicalSegmentationDataset
from env import PixelEnv, NeighborhoodAggregator
from model import PixelDRL_MG
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    test_ds = MedicalSegmentationDataset(args.test_root, split="test",
                                          image_size=cfg.image_size, seed=cfg.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = PixelDRL_MG(in_channels=cfg.in_channels, n_actions=cfg.n_actions,
                         use_sam=cfg.use_sam, use_dc=cfg.use_dc,
                         policy_hidden=cfg.policy_hidden, value_hidden=cfg.value_hidden,
                         n_layers=cfg.n_layers).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    env = PixelEnv(device)
    evaluate(model, env, test_loader, cfg, device, tag="test")


if __name__ == "__main__":
    main()
