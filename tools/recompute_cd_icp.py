"""
Reconcile the a/b CD numbers with the inference-repo benchmark CDs.

The a/b harness computes CD in the raw voxel frame with no alignment; the
inference-repo eval (eval_meshes_paired_emd_voxel.py) optionally normalizes each
mesh and ICP-aligns recon->gt before CD, which absorbs global scale/placement
error. This script recomputes CD for the saved a/b SDFs under the ladder:

    raw            no normalization, no alignment (should reproduce the a/b table)
    icp            best_icp_align (their code) in the raw frame
    norm+icp       normalize_mesh(..., 'bbox_-1_1') on both meshes, then ICP

Uses normalize_mesh / best_icp_align / chamfer_from_dists imported from the
inference repo file itself, so the alignment behavior is theirs exactly.
"""
import os
import sys
import glob
import argparse
import importlib.util

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_metrics import sdf_to_mesh  # noqa: E402

EVAL_PATH = "/projects/gcaddeo/inference/TRELLIS/eval_meshes_paired_emd_voxel.py"
spec = importlib.util.spec_from_file_location("inference_eval", EVAL_PATH)
inference_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_eval)


def sample_pts_nrm(mesh, n, rng):
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(2**31)))
    return np.asarray(pts, np.float64), np.asarray(mesh.face_normals[fi], np.float64)


def cd_pp(rec_pts, gt_pts):
    d_rg = cKDTree(gt_pts).query(rec_pts)[0]
    d_gr = cKDTree(rec_pts).query(gt_pts)[0]
    return inference_eval.chamfer_from_dists(d_rg, d_gr, squared=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf_dir", type=str,
                    default="outputs/diagnostics/ab_guidance_4arm_ycb_sdfs")
    ap.add_argument("--arm", type=str, default="unguided")
    ap.add_argument("--n_points", type=int, default=10000)
    ap.add_argument("--icp_points", type=int, default=3000)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = {"raw": [], "icp": [], "norm+icp": []}
    n_done = 0
    for blob_path in sorted(glob.glob(os.path.join(args.sdf_dir, "samples_*.pt"))):
        blob = torch.load(blob_path, map_location="cpu", weights_only=True)
        k = blob["sdf_gt"].shape[0]
        for b in range(k):
            if args.max_samples > 0 and n_done >= args.max_samples:
                break
            rec = sdf_to_mesh(blob[f"sdf_{args.arm}"][b, 0].numpy())
            gt = sdf_to_mesh(blob["sdf_gt"][b, 0].numpy())
            if rec is None or gt is None:
                continue

            for setting in rows:
                r, g = rec.copy(), gt.copy()
                if setting == "norm+icp":
                    r = inference_eval.normalize_mesh(r, "bbox_-1_1")
                    g = inference_eval.normalize_mesh(g, "bbox_-1_1")
                r_pts, r_nrm = sample_pts_nrm(r, args.n_points, rng)
                g_pts, _ = sample_pts_nrm(g, args.n_points, rng)
                if setting in ("icp", "norm+icp"):
                    sub = rng.choice(len(r_pts), size=min(args.icp_points, len(r_pts)),
                                     replace=False)
                    _, _, _, T, _ = inference_eval.best_icp_align(
                        r, r_pts[sub], r_nrm[sub],
                        g_pts[rng.choice(len(g_pts), size=min(args.icp_points, len(g_pts)),
                                         replace=False)],
                        rng)
                    r_pts = trimesh.transform_points(r_pts, T)
                rows[setting].append(cd_pp(r_pts, g_pts))
            n_done += 1
        if args.max_samples > 0 and n_done >= args.max_samples:
            break

    print(f"arm={args.arm}  n={n_done}")
    for setting, vals in rows.items():
        v = np.asarray(vals)
        print(f"  cd[{setting:>9}] = {v.mean():.5f}  (std {v.std():.4f})")


if __name__ == "__main__":
    main()
