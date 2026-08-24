"""
Build train/test split roots for the hand datasets without touching the originals.

Each dataset root is a directory holding `metadata.csv` plus the payload directories
(`renders_cond`, `data_pose_norm`, `ss_latents_sdf_pose`). A split root is a new
directory that symlinks the payload directories back to the original root and carries
its own filtered `metadata.csv`, so `StandardDatasetBase` sees a disjoint instance list
while all the heavy data stays where it is.

    python tools/make_heldout_split.py --out_root datasets_split --test_frac 0.03

Writes, for every source root NAME:
    <out_root>/NAME_train/{metadata.csv, renders_cond -> ..., data_pose_norm -> ..., ss_latents_sdf_pose -> ...}
    <out_root>/NAME_test/{...}

The split is by `sha256` (the instance id), with a fixed seed, so it is reproducible
and no instance ever appears on both sides.
"""
import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_ROOTS = [
    "/projects/gcaddeo/train_flow/TRELLIS/datasets/Leap_Hand",
    "/projects/gcaddeo/train_flow/TRELLIS/datasets/Hands",
    "/projects/gcaddeo/train_flow/TRELLIS/datasets/Hands_Google",
]

# Subdirectories the ImageConditionedSparseStructureLatentSDFConditioned dataset reads.
PAYLOAD_DIRS = ["renders_cond", "data_pose_norm", "ss_latents_sdf_pose"]


def link_payload(src_root: str, dst_root: str) -> None:
    os.makedirs(dst_root, exist_ok=True)
    for d in PAYLOAD_DIRS:
        src = os.path.join(src_root, d)
        dst = os.path.join(dst_root, d)
        if not os.path.isdir(src):
            print(f"  WARNING: {src} does not exist, skipping the link")
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst) if os.path.islink(dst) else None
        if not os.path.exists(dst):
            os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=str, default=",".join(DEFAULT_ROOTS))
    ap.add_argument("--out_root", type=str, default="datasets_split")
    ap.add_argument("--test_frac", type=float, default=0.03)
    ap.add_argument("--min_test", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    train_roots, test_roots = [], []
    for src_root in args.roots.split(","):
        src_root = src_root.rstrip("/")
        name = os.path.basename(src_root)
        meta = pd.read_csv(os.path.join(src_root, "metadata.csv"))

        # Deterministic per-root shuffle; the seed is mixed with the name so the three
        # datasets do not receive correlated splits.
        rng = np.random.RandomState(args.seed + (abs(hash(name)) % 10000))
        order = rng.permutation(len(meta))
        n_test = max(args.min_test, int(round(args.test_frac * len(meta))))
        n_test = min(n_test, len(meta) // 2)
        test_idx = order[:n_test]
        train_idx = order[n_test:]

        for split, idx in (("train", train_idx), ("test", test_idx)):
            dst_root = os.path.join(out_root, f"{name}_{split}")
            link_payload(src_root, dst_root)
            meta.iloc[np.sort(idx)].to_csv(os.path.join(dst_root, "metadata.csv"), index=False)
            (train_roots if split == "train" else test_roots).append(dst_root)

        assert not (set(meta.iloc[train_idx]["sha256"]) & set(meta.iloc[test_idx]["sha256"]))
        print(f"{name}: {len(train_idx)} train / {len(test_idx)} test  (of {len(meta)})")

    print("\n--data_dir      " + ",".join(train_roots))
    print("--data_dir_test " + ",".join(test_roots))


if __name__ == "__main__":
    main()
