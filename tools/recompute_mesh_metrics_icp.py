"""
Post-hoc mesh metrics WITH ICP alignment for the a/b guidance runs.

Recomputes CD / NC / F@0.02 / EMD for every arm from the SDF blobs saved by
tools/ab_eval_guidance.py, after aligning each reconstructed mesh to the GT
decode with best_icp_align imported from the inference repo's
eval_meshes_paired_emd_voxel.py -- i.e. pose (and whatever else their ICP
absorbs) is factored out, matching how the historical benchmark numbers were
produced. Raw-frame values from the original run remain in the main JSON.

CPU-only. Writes <output> JSON and prints the arm table.

    python tools/recompute_mesh_metrics_icp.py \
        --sdf_dir outputs/diagnostics/ab_guidance_4arm_ycb_sdfs \
        --output  outputs/diagnostics/ab_guidance_4arm_ycb_icp.json
"""
import os
import sys
import glob
import json
import argparse
import importlib.util

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_metrics import sdf_to_mesh, chamfer_from_dists, normal_consistency, \
    fscore_from_dists, emd_l2  # noqa: E402

EVAL_PATH = "/projects/gcaddeo/inference/TRELLIS/eval_meshes_paired_emd_voxel.py"
spec = importlib.util.spec_from_file_location("inference_eval", EVAL_PATH)
inference_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_eval)


def sample_pts_nrm(mesh, n, rng):
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(2**31)))
    return np.asarray(pts, np.float64), np.asarray(mesh.face_normals[fi], np.float64)


def transform_normals(nrm, T):
    R = np.asarray(T)[:3, :3]
    out = nrm @ R.T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


def metrics_aligned(rec_mesh, gt_mesh, rng, n_points, icp_points, tau):
    rec_pts, rec_nrm = sample_pts_nrm(rec_mesh, n_points, rng)
    gt_pts, gt_nrm = sample_pts_nrm(gt_mesh, n_points, rng)

    sub_r = rng.choice(len(rec_pts), size=min(icp_points, len(rec_pts)), replace=False)
    sub_g = rng.choice(len(gt_pts), size=min(icp_points, len(gt_pts)), replace=False)
    _, _, _, T, _ = inference_eval.best_icp_align(
        rec_mesh, rec_pts[sub_r], rec_nrm[sub_r], gt_pts[sub_g], rng)
    rec_pts = trimesh.transform_points(rec_pts, T)
    rec_nrm = transform_normals(rec_nrm, T)

    tree_gt = cKDTree(gt_pts)
    tree_rec = cKDTree(rec_pts)
    d_rg, nn_g = tree_gt.query(rec_pts)
    d_gr, nn_r = tree_rec.query(gt_pts)
    return {
        "cd_icp": chamfer_from_dists(d_rg, d_gr, squared=False),
        "nc_icp": 0.5 * (normal_consistency(rec_nrm, gt_nrm[nn_g])
                         + normal_consistency(gt_nrm, rec_nrm[nn_r])),
        f"f@{tau:g}_icp": fscore_from_dists(d_rg, d_gr, tau=tau),
        "emd_icp": emd_l2(rec_pts, gt_pts, rng, n_points=1024),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf_dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n_points", type=int, default=10000)
    ap.add_argument("--icp_points", type=int, default=3000)
    ap.add_argument("--f_tau", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    acc = {}
    n_done, n_skipped = 0, 0
    for blob_path in sorted(glob.glob(os.path.join(args.sdf_dir, "samples_*.pt"))):
        blob = torch.load(blob_path, map_location="cpu", weights_only=True)
        arms = [k.replace("sdf_", "") for k in blob
                if k.startswith("sdf_") and k not in ("sdf_hand", "sdf_gt")]
        k = blob["sdf_gt"].shape[0]
        for b in range(k):
            gt_mesh = sdf_to_mesh(blob["sdf_gt"][b, 0].numpy())
            if gt_mesh is None:
                n_skipped += 1
                continue
            for arm in arms:
                rec_mesh = sdf_to_mesh(blob[f"sdf_{arm}"][b, 0].numpy())
                if rec_mesh is None:
                    continue
                m = metrics_aligned(rec_mesh, gt_mesh, rng,
                                    args.n_points, args.icp_points, args.f_tau)
                for kk, vv in m.items():
                    acc.setdefault(arm, {}).setdefault(kk, []).append(vv)
            n_done += 1
            if n_done % 8 == 0:
                print(f"  ... {n_done} samples")

    results = {"meta": {"sdf_dir": args.sdf_dir, "n_samples": n_done,
                        "n_skipped": n_skipped, "icp_points": args.icp_points,
                        "n_points": args.n_points, "seed": args.seed}}
    for arm in acc:
        results[arm] = {kk: {"mean": float(np.mean(vv)), "std": float(np.std(vv)),
                             "n": len(vv)}
                        for kk, vv in acc[arm].items()}

    arms = sorted(acc.keys())
    metrics = sorted(next(iter(acc.values())).keys())
    print(f"\n{'metric':<12}" + "".join(f"{a:>16}" for a in arms))
    for kk in metrics:
        print(f"{kk:<12}" + "".join(f"{results[a][kk]['mean']:>16.5g}" for a in arms))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
