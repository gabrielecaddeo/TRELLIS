#!/usr/bin/env python3
# check_pose_features_plus.py
#
# Adds:
#  - A vs A sanity check
#  - One-way and mutual NN diagnostics
#  - Thresholded overlap@thr metrics in world space
#  - Comparison against vanilla features/<model>/<sha>.npz (gold)
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

def summarize_cos(name, cos):
    return {
        f"{name}_mean": float(cos.mean()),
        f"{name}_median": float(np.median(cos)),
        f"{name}_p10": float(np.percentile(cos, 10)),
        f"{name}_p90": float(np.percentile(cos, 90)),
    }

def summarize_dist(name, dist):
    return {
        f"{name}_mean": float(dist.mean()),
        f"{name}_p50": float(np.percentile(dist, 50)),
        f"{name}_p90": float(np.percentile(dist, 90)),
        f"{name}_p99": float(np.percentile(dist, 99)),
    }


# -----------------------------
# Utilities: geometry transforms (pose_norm)
# -----------------------------
def idxs_to_centers_unit2(idxs, origin, voxel_size):
    idxs = idxs.astype(np.float32)
    origin = np.array(origin, dtype=np.float32)
    centers = origin + (idxs + 0.5) * float(voxel_size)
    return centers

def posed_to_canonical_row(x_pose, R_row, s, t, c0):
    x = (x_pose - t[None, :]) / float(s)
    x = x @ R_row.T
    x = x + c0[None, :]
    return x

def canonical_to_render_world(x_can, s_norm, c_norm):
    return (x_can / float(s_norm)) + c_norm[None, :]

def pose_centers_world(meta_path, idx_path):
    posed_meta = json.load(open(meta_path, "r"))
    idxs = np.load(idx_path).astype(np.int32)

    grid = posed_meta["grid"]
    origin = tuple(grid["origin"])
    voxel_size = float(grid["voxel_size"])

    s_norm = float(posed_meta["canonical"]["s_norm"])
    c_norm = np.array(posed_meta["canonical"]["c_norm"], dtype=np.float32)

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
# Vanilla voxel centers from indices
# -----------------------------
def vanilla_idxs_to_centers_world(idxs_ijk, res=64):
    """
    Vanilla script did:
        indices = ((positions + 0.5) * 64).long()
    => positions = indices/64 - 0.5
    These "positions" are in the render/world frame of blender pipeline.
    """
    idxs = idxs_ijk.astype(np.float32)
    pos = idxs / float(res) - 0.5
    return pos.astype(np.float32)


# -----------------------------
# Matching
# -----------------------------
def nn_match(A_xyz, B_xyz):
    if HAVE_KDTREE:
        tree = KDTree(B_xyz)
        dist, nn = tree.query(A_xyz, k=1)
        return dist.astype(np.float32), nn.astype(np.int64)
    # brute force fallback
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
    A_idx = np.arange(nnAB.shape[0])
    return (nnBA[nnAB] == A_idx)

def uniqueness_ratio(nn_idx):
    return float(len(np.unique(nn_idx)) / max(1, nn_idx.shape[0]))


# -----------------------------
# IO helpers
# -----------------------------
def load_pose_npz(features_root, pose_tag):
    path = os.path.join(features_root, f"{pose_tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pack = np.load(path)
    feats = pack["patchtokens"].astype(np.float32)
    idxs  = pack["indices"].astype(np.int32)
    count = pack["count"].astype(np.int32) if "count" in pack.files else None
    return feats, idxs, count, path

def load_vanilla_npz(features_root, sha):
    """
    Vanilla gold file: <sha>.npz (no _fXXX)
    Must contain:
      - indices [N,3] uint8
      - patchtokens [N,1024] float16/float32
    """
    path = os.path.join(features_root, f"{sha}.npz")
    if not os.path.exists(path):
        return None
    pack = np.load(path)
    feats = pack["patchtokens"].astype(np.float32)
    idxs  = pack["indices"].astype(np.int32)
    count = pack["count"].astype(np.int32) if "count" in pack.files else None
    return feats, idxs, count, path

def find_pose_files(root, sha, model):
    pose_dir = os.path.join(root, "data_pose_norm", sha)
    feat_dir = os.path.join(root, "features", model)
    meta_paths = sorted(glob.glob(os.path.join(pose_dir, f"{sha}_f*_meta.json")))
    tags = [os.path.basename(p).replace("_meta.json", "") for p in meta_paths]
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

    print("norm stats:", feat_norm_stats(feats))
    print("random cosine (raw):", rand_cos_stats(feats, n_pairs=rand_pairs, seed=seed))

    feats_c, _ = mean_center(feats)
    feats_cn = l2norm_rows(feats_c)
    print("random cosine (centered+L2):", rand_cos_stats(feats_cn, n_pairs=rand_pairs, seed=seed))

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


def compute_pair_metrics(
    tagA, featsA, countA, A_xyz,
    tagB, featsB, countB, B_xyz,
    min_count, mutual=True, thr=None
):
    """
    Returns metrics dict for:
      - one-way NN (A->B): always computed
      - mutual NN: computed if mutual=True
      - thresholded overlap@thr: computed if thr not None
    """
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

    distAB, nnAB = nn_match(A_xyz_f, B_xyz_f)  # one-way

    # threshold mask (geometric overlap)
    thr_mask = np.ones_like(distAB, dtype=bool)
    if thr is not None:
        thr_mask = distAB <= float(thr)

    # mutual mask
    if mutual:
        distBA, nnBA = nn_match(B_xyz_f, A_xyz_f)
        m_mask = mutual_nn_mask(nnAB, nnBA)
    else:
        m_mask = np.ones_like(nnAB, dtype=bool)

    # combine masks for "matched"
    matched_mask = m_mask & thr_mask
    matched = int(matched_mask.sum())
    if matched == 0:
        return {
            "poseA": tagA, "poseB": tagB,
            "usedA": int(featsA_f.shape[0]), "usedB": int(featsB_f.shape[0]),
            "matched": 0,
            "match_rate_A": 0.0,
            "overlap_A@thr": float(thr_mask.mean()) if thr is not None else np.nan,
            "distAB_mean": float(distAB.mean()),
        }

    a = featsA_f[matched_mask]
    b = featsB_f[nnAB[matched_mask]]

    cos_raw = cosine_rows(a, b)
    a_cn = l2norm_rows(mean_center(featsA_f)[0])[matched_mask]
    b_cn = l2norm_rows(mean_center(featsB_f)[0])[nnAB[matched_mask]]
    cos_cn = cosine_rows(a_cn, b_cn)

    out = {
        "poseA": tagA,
        "poseB": tagB,
        "totalA": int(featsA.shape[0]),
        "totalB": int(featsB.shape[0]),
        "usedA": int(featsA_f.shape[0]),
        "usedB": int(featsB_f.shape[0]),
        "matched": matched,
        "mutual": bool(mutual),
        "thr": float(thr) if thr is not None else np.nan,
        "match_rate_A": float(matched / max(1, featsA_f.shape[0])),
        "uniq_ratio_B": uniqueness_ratio(nnAB[matched_mask]),
        "overlap_A@thr": float(thr_mask.mean()) if thr is not None else np.nan,
    }
    out.update(summarize_dist("distAB", distAB[matched_mask]))
    out.update(summarize_cos("cos_raw", cos_raw))
    out.update(summarize_cos("cos_cn", cos_cn))
    return out


def print_row_short(out):
    if out is None:
        return
    thr_str = "" if np.isnan(out.get("thr", np.nan)) else f" thr={out['thr']:.4f}"
    ov_str = "" if np.isnan(out.get("overlap_A@thr", np.nan)) else f" ovA@thr={out['overlap_A@thr']:.3f}"
    print(
        f"{out['poseB']}: matched={out['matched']} "
        f"match_rate_A={out['match_rate_A']:.3f} uniqB={out.get('uniq_ratio_B',np.nan):.3f} "
        f"distAB_mean={out.get('distAB_mean',np.nan):.4f} "
        f"cos_raw_mean={out.get('cos_raw_mean',np.nan):.4f} "
        f"cos_cn_mean={out.get('cos_cn_mean',np.nan):.4f}"
        f"{thr_str}{ov_str}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--sha", type=str, required=True)
    ap.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    ap.add_argument("--pose_tag", type=str, default=None)
    ap.add_argument("--min_count", type=int, default=8)
    ap.add_argument("--max_poses", type=int, default=24)
    ap.add_argument("--rand_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--no_mutual", dest="mutual", action="store_false")
    ap.set_defaults(mutual=True)

    # NEW knobs
    ap.add_argument("--thr_mult", type=float, default=1,
                    help="threshold = thr_mult * world_voxel_size (estimated). Set 0 to disable thresholding.")
    ap.add_argument("--world_voxel_size", type=float, default=None,
                    help="Override world voxel size directly. If omitted, we estimate from vanilla grid: 1/res.")
    ap.add_argument("--vanilla_res", type=int, default=64, help="vanilla grid res for sha.npz positions")
    ap.add_argument("--save_csv", type=str, default=None)

    args = ap.parse_args()

    if not HAVE_KDTREE:
        print("[WARN] scipy not found: brute-force matching will be slow. Install scipy for speed.")

    root = args.root
    sha = args.sha
    model = args.model

    pose_dir, feat_dir, tags = find_pose_files(root, sha, model)
    if len(tags) == 0:
        raise RuntimeError(f"No pose-norm features found for sha={sha} in {feat_dir}")

    pose_tag = args.pose_tag or tags[0]
    if pose_tag not in tags:
        raise RuntimeError(f"pose_tag {pose_tag} not found. Available: {tags[:10]}{'...' if len(tags)>10 else ''}")

    # Vanilla gold
    vanilla = load_vanilla_npz(feat_dir, sha)

    # Choose threshold
    # If you use vanilla world frame in [-0.5,0.5]^3, voxel size is 1/res.
    if args.world_voxel_size is not None:
        w_dx = float(args.world_voxel_size)
    else:
        w_dx = 1.0 / float(args.vanilla_res)

    thr = None
    if args.thr_mult and args.thr_mult > 0:
        thr = args.thr_mult * w_dx

    # ---- Load pose A ----
    featsA, idxA, countA, npzA = load_pose_npz(feat_dir, pose_tag)
    metaA_path, idxA_path = meta_and_idx_paths(pose_dir, pose_tag)
    _, A_xyz, _ = pose_centers_world(metaA_path, idxA_path)

    print(f"Pose A: {pose_tag}")
    print(f"  npz:  {npzA}")
    print(f"  meta: {metaA_path}")
    print(f"  idx:  {idxA_path}")
    print(f"  kd-tree: {HAVE_KDTREE}")
    print(f"  world_voxel_size={w_dx:.6f}  thr_mult={args.thr_mult}  thr={thr if thr is not None else 'disabled'}")
    if vanilla is not None:
        print(f"  vanilla gold: {vanilla[3]}")
    else:
        print(f"  vanilla gold: NOT FOUND at {os.path.join(feat_dir, sha + '.npz')}")
    print()

    # single pose report
    print_single_pose_report(featsA, countA, args.min_count, args.rand_pairs, args.seed)

    # ---- A vs A sanity check ----
    print("=== Sanity: Pose A vs Pose A ===")
    outAA = compute_pair_metrics(
        pose_tag, featsA, countA, A_xyz,
        pose_tag, featsA, countA, A_xyz,
        min_count=args.min_count,
        mutual=args.mutual,
        thr=thr
    )
    print_row_short(outAA)
    print("Expected: match_rate_A ~ 1.0, distAB_mean ~ 0.0, cos_cn_mean ~ 1.0\n")

    # ---- Compare to other poses ----
    others = [t for t in tags if t != pose_tag]
    if args.max_poses is not None:
        others = others[:args.max_poses]

    rows = []
    print("=== Cross-pose comparisons (pose-norm) ===")
    print(f"Comparing against {len(others)} poses (mutual={args.mutual}, min_count={args.min_count})\n")

    for poseB in others:
        featsB, idxB, countB, _ = load_pose_npz(feat_dir, poseB)
        metaB_path, idxB_path = meta_and_idx_paths(pose_dir, poseB)
        _, B_xyz, _ = pose_centers_world(metaB_path, idxB_path)

        out = compute_pair_metrics(
            pose_tag, featsA, countA, A_xyz,
            poseB, featsB, countB, B_xyz,
            min_count=args.min_count,
            mutual=args.mutual,
            thr=thr
        )
        rows.append(out)
        print_row_short(out)

    # ---- Compare Pose A to Vanilla Gold ----
    if vanilla is not None:
        featsV, idxV, countV, pathV = vanilla
        V_xyz = vanilla_idxs_to_centers_world(idxV, res=args.vanilla_res)

        print("\n=== Compare Pose A to vanilla gold ===")
        outAV = compute_pair_metrics(
            pose_tag, featsA, countA, A_xyz,
            sha, featsV, countV, V_xyz,
            min_count=args.min_count,
            mutual=args.mutual,
            thr=thr
        )
        print_row_short(outAV)

        # Also print one-way non-mutual (often more informative with density differences)
        outAV_oneway = compute_pair_metrics(
            pose_tag, featsA, countA, A_xyz,
            sha, featsV, countV, V_xyz,
            min_count=args.min_count,
            mutual=False,
            thr=thr
        )
        print("One-way A->vanilla (mutual=False):")
        print_row_short(outAV_oneway)

    # summarize
    def safe_arr(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=np.float32)

    if rows:
        cos_cn = safe_arr("cos_cn_mean")
        cos_raw = safe_arr("cos_raw_mean")
        dist_m  = safe_arr("distAB_mean")
        mr      = safe_arr("match_rate_A")
        ov      = safe_arr("overlap_A@thr") if thr is not None else None

        print("\n=== Summary (pose-norm A vs other poses) ===")
        print(f"cos_cn_mean  : mean={np.nanmean(cos_cn):.4f} p10={np.nanpercentile(cos_cn,10):.4f} p90={np.nanpercentile(cos_cn,90):.4f}")
        print(f"cos_raw_mean : mean={np.nanmean(cos_raw):.4f} p10={np.nanpercentile(cos_raw,10):.4f} p90={np.nanpercentile(cos_raw,90):.4f}")
        print(f"distAB_mean  : mean={np.nanmean(dist_m):.4f} p10={np.nanpercentile(dist_m,10):.4f} p90={np.nanpercentile(dist_m,90):.4f}")
        print(f"match_rate_A : mean={np.nanmean(mr):.4f} p10={np.nanpercentile(mr,10):.4f} p90={np.nanpercentile(mr,90):.4f}")
        if ov is not None:
            print(f"overlap_A@thr: mean={np.nanmean(ov):.4f} p10={np.nanpercentile(ov,10):.4f} p90={np.nanpercentile(ov,90):.4f}")
    else:
        print("No comparison rows produced.")

    # optional CSV
    if args.save_csv and rows:
        import csv
        keys = sorted({k for r in rows for k in r.keys()})
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
