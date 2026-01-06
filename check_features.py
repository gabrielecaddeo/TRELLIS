#!/usr/bin/env python3
# check_pose_features.py
#
# Usage:
#   python check_pose_features.py --root /path/to/dataset \
#       --sha 005b50cba077963f... \
#       --model dinov2_vitl14_reg \
#       --pose_tag 005b50cba077963f..._f015 \
#       --min_count 8 \
#       --k 1 \
#       --max_poses 24 \
#       --rand_pairs 20000
#
# What it does:
#   1) Loads ONE pose .npz and reports:
#        - feature norms distribution
#        - random cosine similarity distribution (raw)
#        - random cosine distribution after mean-centering + L2 normalize
#        - same stats but only for voxels with count >= min_count
#
#   2) Compares that pose to other poses of the same object:
#        - builds world-space centers for BOTH poses using meta + idxs
#        - matches points by nearest neighbor in world space (k=1 default)
#        - reports cosine stats on matched pairs:
#            raw cosine, centered+L2 cosine
#        - reports mutual-NN match rate (strictness) and uniqueness ratio
#
# Notes:
#   - Requires only numpy + scipy. If scipy is not available, it falls back to
#     a slower brute-force matcher (OK for small N; not recommended).
#
import os
import glob
import json
import argparse
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False


# -----------------------------
# Utilities: features
# -----------------------------
def l2norm_rows(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def mean_center(x):
    mu = x.mean(axis=0, keepdims=True)
    return x - mu, mu

def cosine_rows(a, b, eps=1e-8):
    # a,b: [M,C]
    na = np.linalg.norm(a, axis=1) + eps
    nb = np.linalg.norm(b, axis=1) + eps
    return (a * b).sum(axis=1) / (na * nb)

def rand_cos_stats(feats, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    N = feats.shape[0]
    if N < 2:
        return {"cos_rand_mean": np.nan, "cos_rand_p10": np.nan, "cos_rand_p90": np.nan}
    i = rng.integers(0, N, size=n_pairs, endpoint=False)
    j = rng.integers(0, N, size=n_pairs, endpoint=False)
    cos = cosine_rows(feats[i], feats[j])
    return {
        "cos_rand_mean": float(cos.mean()),
        "cos_rand_p10": float(np.percentile(cos, 10)),
        "cos_rand_p90": float(np.percentile(cos, 90)),
    }

def feat_norm_stats(feats):
    n = np.linalg.norm(feats, axis=1)
    return {
        "norm_mean": float(n.mean()),
        "norm_std": float(n.std()),
        "norm_p10": float(np.percentile(n, 10)),
        "norm_p50": float(np.percentile(n, 50)),
        "norm_p90": float(np.percentile(n, 90)),
    }


# -----------------------------
# Utilities: geometry transforms
# -----------------------------
def idxs_to_centers_unit2(idxs, origin, voxel_size):
    # idxs: (N,3) int
    idxs = idxs.astype(np.float32)
    origin = np.array(origin, dtype=np.float32)
    centers = origin + (idxs + 0.5) * float(voxel_size)
    return centers  # (N,3) float32

def posed_to_canonical_row(x_pose, R_row, s, t, c0):
    # x_pose = ((x_can - c0) @ R) * s + t  (row convention)
    # => x_can = ((x_pose - t)/s) @ R^T + c0
    x = (x_pose - t[None, :]) / float(s)
    x = x @ R_row.T
    x = x + c0[None, :]
    return x

def canonical_to_render_world(x_can, s_norm, c_norm):
    # x_can = (x_raw - c_norm) * s_norm  => x_raw = x_can / s_norm + c_norm
    return (x_can / float(s_norm)) + c_norm[None, :]

def pose_centers_world(meta_path, idx_path):
    posed_meta = json.load(open(meta_path, "r"))
    idxs = np.load(idx_path).astype(np.int32)

    # grid
    grid = posed_meta["grid"]
    origin = tuple(grid["origin"])
    voxel_size = float(grid["voxel_size"])

    # canonical norm (saved by you)
    s_norm = float(posed_meta["canonical"]["s_norm"])
    c_norm = np.array(posed_meta["canonical"]["c_norm"], dtype=np.float32)

    # pose params
    pose = posed_meta["pose"]
    R_row = np.array(pose["R_fixed"], dtype=np.float32)
    s_aug = float(pose["s_aug"])
    t_aug = np.array(pose["t_aug"], dtype=np.float32)
    c0    = np.array(pose["c0"], dtype=np.float32)

    centers_pose = idxs_to_centers_unit2(idxs, origin=origin, voxel_size=voxel_size).astype(np.float32)
    centers_can  = posed_to_canonical_row(centers_pose, R_row, s_aug, t_aug, c0).astype(np.float32)
    centers_w    = canonical_to_render_world(centers_can, s_norm, c_norm).astype(np.float32)
    return idxs, centers_w, posed_meta


# -----------------------------
# Matching
# -----------------------------
def nn_match(A_xyz, B_xyz, k=1):
    """
    Returns:
      dist: (NA,) nearest distance
      nn:   (NA,) index in B for each A
    """
    if HAVE_KDTREE:
        tree = KDTree(B_xyz)
        dist, nn = tree.query(A_xyz, k=k)
        if k != 1:
            # keep best
            dist = dist[:, 0]
            nn = nn[:, 0]
        return dist.astype(np.float32), nn.astype(np.int64)

    # fallback brute force (slow)
    NA = A_xyz.shape[0]
    nn = np.zeros((NA,), dtype=np.int64)
    dist = np.zeros((NA,), dtype=np.float32)
    for i in range(NA):
        d2 = ((B_xyz - A_xyz[i]) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        nn[i] = j
        dist[i] = float(np.sqrt(d2[j]))
    return dist, nn

def mutual_nn_mask(nnAB, nnBA):
    # nnAB: for each A gives B index
    # nnBA: for each B gives A index
    A_idx = np.arange(nnAB.shape[0])
    return (nnBA[nnAB] == A_idx)

def uniqueness_ratio(nn_idx):
    # fraction of unique targets in B
    return float(len(np.unique(nn_idx)) / max(1, nn_idx.shape[0]))


# -----------------------------
# IO helpers
# -----------------------------
def load_pose_npz(features_root, pose_tag):
    path = os.path.join(features_root, f"{pose_tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pack = np.load(path)
    feats = pack["patchtokens"].astype(np.float32)  # [N,C]
    idxs  = pack["indices"].astype(np.int32)        # [N,3]
    count = pack["count"].astype(np.int32) if "count" in pack.files else None
    return feats, idxs, count, path

def find_pose_files(root, sha, model):
    pose_dir = os.path.join(root, "data_pose_norm", sha)
    feat_dir = os.path.join(root, "features", model)
    meta_paths = sorted(glob.glob(os.path.join(pose_dir, f"{sha}_f*_meta.json")))
    # pose_tag from meta
    tags = [os.path.basename(p).replace("_meta.json", "") for p in meta_paths]
    # only keep those that have npz
    tags = [t for t in tags if os.path.exists(os.path.join(feat_dir, f"{t}.npz"))]
    return pose_dir, feat_dir, tags

def meta_and_idx_paths(pose_dir, pose_tag):
    meta_path = os.path.join(pose_dir, f"{pose_tag}_meta.json")
    idx_path  = os.path.join(pose_dir, "idxs", f"{pose_tag}.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    if not os.path.exists(idx_path):
        raise FileNotFoundError(idx_path)
    return meta_path, idx_path


# -----------------------------
# Reports
# -----------------------------
def print_single_pose_report(feats, count, min_count, rand_pairs, seed):
    print("=== Single pose checks ===")
    print(f"feats: {feats.shape} dtype={feats.dtype}")

    # norms
    ns = feat_norm_stats(feats)
    print("norm stats:", ns)

    # random cosine raw
    rc = rand_cos_stats(feats, n_pairs=rand_pairs, seed=seed)
    print("random cosine (raw):", rc)

    # centered + L2
    feats_c, _ = mean_center(feats)
    feats_cn = l2norm_rows(feats_c)
    rc2 = rand_cos_stats(feats_cn, n_pairs=rand_pairs, seed=seed)
    print("random cosine (centered+L2):", rc2)

    if count is not None:
        mask = count >= min_count
        print(f"count>= {min_count}: {int(mask.sum())}/{len(count)} ({100*mask.mean():.1f}%)")
        if mask.sum() >= 2:
            feats_g = feats[mask]
            print("  norm stats (good):", feat_norm_stats(feats_g))
            print("  random cosine raw (good):", rand_cos_stats(feats_g, n_pairs=rand_pairs, seed=seed))
            feats_gcn = l2norm_rows(mean_center(feats_g)[0])
            print("  random cosine centered+L2 (good):", rand_cos_stats(feats_gcn, n_pairs=rand_pairs, seed=seed))
    print()


def compare_two_poses(
    poseA_tag, featsA, countA, A_xyz,
    poseB_tag, featsB, countB, B_xyz,
    min_count, mutual=True
):
    # filter by count first (optional but recommended)
    maskA = np.ones((featsA.shape[0],), dtype=bool)
    maskB = np.ones((featsB.shape[0],), dtype=bool)
    if countA is not None:
        maskA &= (countA >= min_count)
    if countB is not None:
        maskB &= (countB >= min_count)

    A_xyz_f = A_xyz[maskA]
    B_xyz_f = B_xyz[maskB]
    featsA_f = featsA[maskA]
    featsB_f = featsB[maskB]

    if featsA_f.shape[0] == 0 or featsB_f.shape[0] == 0:
        return None

    # NN match by world position
    distAB, nnAB = nn_match(A_xyz_f, B_xyz_f, k=1)

    if mutual:
        distBA, nnBA = nn_match(B_xyz_f, A_xyz_f, k=1)
        m = mutual_nn_mask(nnAB, nnBA)
    else:
        m = np.ones_like(nnAB, dtype=bool)

    matched = int(m.sum())
    if matched == 0:
        return None

    # cosines
    a = featsA_f[m]
    b = featsB_f[nnAB[m]]

    cos_raw = cosine_rows(a, b)
    # centered+L2 per-pose (important!)
    a_cn = l2norm_rows(mean_center(featsA_f)[0])[m]
    b_cn = l2norm_rows(mean_center(featsB_f)[0])[nnAB[m]]
    cos_cn = cosine_rows(a_cn, b_cn)

    out = {
        "poseA": poseA_tag,
        "poseB": poseB_tag,
        "totalA": int(featsA.shape[0]),
        "totalB": int(featsB.shape[0]),
        "usedA": int(featsA_f.shape[0]),
        "usedB": int(featsB_f.shape[0]),
        "matched": matched,
        "mutual": bool(mutual),
        "match_rate_A": float(matched / max(1, featsA_f.shape[0])),
        "uniq_ratio_B": uniqueness_ratio(nnAB[m]),
        "dist_mean": float(distAB[m].mean()),
        "dist_p90": float(np.percentile(distAB[m], 90)),
        "cos_raw_mean": float(cos_raw.mean()),
        "cos_raw_median": float(np.median(cos_raw)),
        "cos_raw_p10": float(np.percentile(cos_raw, 10)),
        "cos_raw_p90": float(np.percentile(cos_raw, 90)),
        "cos_cn_mean": float(cos_cn.mean()),
        "cos_cn_median": float(np.median(cos_cn)),
        "cos_cn_p10": float(np.percentile(cos_cn, 10)),
        "cos_cn_p90": float(np.percentile(cos_cn, 90)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root containing data_pose_norm/ and features/")
    ap.add_argument("--sha", type=str, required=True, help="object sha256 folder name")
    ap.add_argument("--model", type=str, default="dinov2_vitl14_reg", help="features/<model>/")
    ap.add_argument("--pose_tag", type=str, default=None, help="pose tag like <sha>_f015. If omitted, use first available.")
    ap.add_argument("--min_count", type=int, default=8)
    ap.add_argument("--max_poses", type=int, default=24, help="compare against up to this many poses")
    ap.add_argument("--rand_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mutual", action="store_true", help="use mutual NN matching (recommended)")
    ap.add_argument("--no_mutual", dest="mutual", action="store_false")
    ap.set_defaults(mutual=True)
    ap.add_argument("--save_csv", type=str, default=None, help="optional path to save comparison rows as CSV")
    args = ap.parse_args()

    root = args.root
    sha = args.sha
    model = args.model

    if not HAVE_KDTREE:
        print("[WARN] scipy not found: falling back to brute-force matching (slow). Install scipy for speed.")

    pose_dir, feat_dir, tags = find_pose_files(root, sha, model)
    if len(tags) == 0:
        raise RuntimeError(f"No poses with features found for sha={sha} in {feat_dir}")

    pose_tag = args.pose_tag or tags[0]
    if pose_tag not in tags:
        raise RuntimeError(f"pose_tag {pose_tag} not found. Available: {tags[:10]}{'...' if len(tags)>10 else ''}")

    # ---- Load pose A ----
    featsA, idxA, countA, npzA = load_pose_npz(feat_dir, pose_tag)
    metaA_path, idxA_path = meta_and_idx_paths(pose_dir, pose_tag)
    _, A_xyz, _ = pose_centers_world(metaA_path, idxA_path)

    print(f"Pose A: {pose_tag}")
    print(f"  npz:  {npzA}")
    print(f"  meta: {metaA_path}")
    print(f"  idx:  {idxA_path}")
    print()

    # single pose report
    print_single_pose_report(featsA, countA, args.min_count, args.rand_pairs, args.seed)

    # ---- Compare to other poses ----
    others = [t for t in tags if t != pose_tag]
    if args.max_poses is not None:
        others = others[:args.max_poses]

    rows = []
    print("=== Cross-pose comparisons ===")
    print(f"Comparing against {len(others)} poses (mutual={args.mutual}, min_count={args.min_count})")
    print()

    for poseB in others:
        featsB, idxB, countB, _ = load_pose_npz(feat_dir, poseB)
        metaB_path, idxB_path = meta_and_idx_paths(pose_dir, poseB)
        _, B_xyz, _ = pose_centers_world(metaB_path, idxB_path)

        out = compare_two_poses(
            pose_tag, featsA, countA, A_xyz,
            poseB, featsB, countB, B_xyz,
            min_count=args.min_count,
            mutual=args.mutual,
        )
        if out is None:
            print(f"{poseB}: no matches (after filtering) or empty.")
            continue

        rows.append(out)
        print(
            f"{poseB}: matched={out['matched']} "
            f"match_rate_A={out['match_rate_A']:.3f} uniqB={out['uniq_ratio_B']:.3f} "
            f"dist_mean={out['dist_mean']:.4f} "
            f"cos_raw_mean={out['cos_raw_mean']:.4f} "
            f"cos_cn_mean={out['cos_cn_mean']:.4f}"
        )

    # summarize
    if rows:
        cos_raw = np.array([r["cos_raw_mean"] for r in rows], dtype=np.float32)
        cos_cn  = np.array([r["cos_cn_mean"] for r in rows], dtype=np.float32)
        dist_m  = np.array([r["dist_mean"] for r in rows], dtype=np.float32)
        mr      = np.array([r["match_rate_A"] for r in rows], dtype=np.float32)
        print()
        print("=== Summary ===")
        print(f"cos_raw_mean over poses: mean={cos_raw.mean():.4f} p10={np.percentile(cos_raw,10):.4f} p90={np.percentile(cos_raw,90):.4f}")
        print(f"cos_cn_mean  over poses: mean={cos_cn.mean():.4f} p10={np.percentile(cos_cn,10):.4f} p90={np.percentile(cos_cn,90):.4f}")
        print(f"dist_mean    over poses: mean={dist_m.mean():.4f} p10={np.percentile(dist_m,10):.4f} p90={np.percentile(dist_m,90):.4f}")
        print(f"match_rate_A over poses: mean={mr.mean():.4f} p10={np.percentile(mr,10):.4f} p90={np.percentile(mr,90):.4f}")
    else:
        print("No comparison rows produced.")

    # optional CSV
    if args.save_csv and rows:
        import csv
        keys = list(rows[0].keys())
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
