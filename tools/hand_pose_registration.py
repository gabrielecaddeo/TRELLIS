"""Hand-volume similarity registration (user idea, 2026-09-02): recover the
RELATIVE pose between two views of a grasp from the hand CONDITIONING volumes
alone — no metas, no camera extrinsics.

Within a grasp the hand is a rigid body and each view carries its COMPLETE
hand SDF in its own grid (conditioning from FK, not camera-dependent), so
registering the two volumes is full-shape similarity registration:
  x_src = a * R @ x_dst + t          (the same family as the meta warps)
- a from the interior-volume ratio (V scales as a^3),
- R init from PCA/inertia axes (4 det+1 sign combos, best initial cost),
- t from centroids,
- refinement: least-squares on SDF residuals sdf_src(a R x + t)/a over the
  dst near-surface band (correspondence-free; trilinear sampling).

`estimate_similarity` returns (a, R, t, diag); `warp_sdf_affine` resamples a
volume through the estimated map (values rescaled to dst units, same
convention as multiview_warp.warp_sdf). Run as a script to VALIDATE against
the GT metas on held-out instances (angle/translation/scale errors + the
object-warp IoU achieved with estimated vs GT poses).
"""
import os, sys, json, argparse
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiview_warp import load_view, sim_from_meta, warp_sdf, sdf_agreement  # noqa: E402

RES = 64
D = 2.0 / RES


def _grid_pts(mask):
    idx = np.argwhere(mask)
    return -1.0 + (idx + 0.5) * D


def _sample(sdf, pts, cval=1.0):
    idx = (pts + 1.0) / D - 0.5
    return map_coordinates(sdf, idx.T, order=1, mode="constant", cval=cval)


def _rot_from_axis_angle(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    ax = w / th
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def estimate_similarity(sdf_src, sdf_dst, band=0.05, n_surf=4000, seed=0):
    """Estimate (a, R, t) with x_src = a*R@x_dst + t from two hand SDFs."""
    rng = np.random.default_rng(seed)
    in_s, in_d = sdf_src < 0, sdf_dst < 0
    if in_s.sum() < 50 or in_d.sum() < 50:
        return None
    a = (in_s.sum() / in_d.sum()) ** (1.0 / 3.0)

    P_s, P_d = _grid_pts(in_s), _grid_pts(in_d)
    c_s, c_d = P_s.mean(0), P_d.mean(0)
    # PCA axes (interior points), 4 proper sign combos
    _, V_s = np.linalg.eigh(np.cov((P_s - c_s).T))
    _, V_d = np.linalg.eigh(np.cov((P_d - c_d).T))
    surf = _grid_pts(np.abs(sdf_dst) < band)
    if len(surf) > n_surf:
        surf = surf[rng.choice(len(surf), n_surf, replace=False)]

    best = None
    for signs in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        R0 = V_s @ np.diag(signs) @ V_d.T
        if np.linalg.det(R0) < 0:
            continue
        t0 = c_s - a * (R0 @ c_d)
        cost = np.mean(np.abs(_sample(sdf_src, (a * (R0 @ surf.T)).T + t0) / a))
        if best is None or cost < best[0]:
            best = (cost, R0, t0)
    _, R0, t0 = best

    def resid(p):
        R = _rot_from_axis_angle(p[:3]) @ R0
        t = p[3:6]
        aa = a * np.exp(p[6])
        return _sample(sdf_src, (aa * (R @ surf.T)).T + t) / aa

    sol = least_squares(resid, np.concatenate([np.zeros(3), t0, [0.0]]),
                        method="trf", diff_step=1e-3, max_nfev=60)
    R = _rot_from_axis_angle(sol.x[:3]) @ R0
    t = sol.x[3:6]
    a_fin = a * np.exp(sol.x[6])
    return a_fin, R, t, {"cost0": float(best[0]),
                         "cost": float(np.mean(np.abs(sol.fun))),
                         "n_surf": len(surf)}


def warp_sdf_affine(sdf_src, a, R, t, cval=1.0):
    """Resample sdf_src onto the dst grid through x_src = a*R@x_dst + t
    (values rescaled by 1/a into dst units)."""
    ax = -1.0 + (np.arange(RES) + 0.5) * D
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts_dst = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    pts_src = (a * (R @ pts_dst.T)).T + t
    idx = (pts_src + 1.0) / D - 0.5
    out = map_coordinates(sdf_src, idx.T, order=1, mode="constant", cval=cval * a)
    return (out / a).reshape(RES, RES, RES)


def gt_relative(meta_src, meta_dst):
    """GT (a, R, t) with x_src = a*R@x_dst + t, composed from the metas."""
    s_s, R_s, t_s, _ = sim_from_meta(meta_src)
    s_d, R_d, t_d, _ = sim_from_meta(meta_dst)
    # x_canon = R_d @ (x_dst - t_d)/s_d ; x_src = s_s * R_s.T @ x_canon + t_s
    a = s_s / s_d
    R = R_s.T @ R_d
    t = t_s - a * (R @ t_d)
    return a, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="one split root (uses data_pose_norm)")
    ap.add_argument("--num_instances", type=int, default=12)
    ap.add_argument("--pairs", default="3-0,6-0,12-0,21-0,9-15")
    args = ap.parse_args()

    inst_root = os.path.join(args.data_dir, "data_pose_norm")
    instances = sorted(os.listdir(inst_root))[:args.num_instances]
    pairs = [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")]

    errs = {"deg": [], "t": [], "s": [], "iou_est": [], "iou_gt": []}
    for inst in instances:
        d = os.path.join(inst_root, inst)
        for i, j in pairs:
            try:
                m_i, s_i = load_view(d, inst, i)
                m_j, s_j = load_view(d, inst, j)
            except Exception:
                continue
            if "hand" not in s_i or "hand" not in s_j or "object" not in s_i:
                continue
            est = estimate_similarity(s_i["hand"], s_j["hand"])
            if est is None:
                continue
            a_e, R_e, t_e, diag = est
            a_g, R_g, t_g = gt_relative(m_i, m_j)
            ang = np.degrees(np.arccos(np.clip((np.trace(R_e @ R_g.T) - 1) / 2, -1, 1)))
            errs["deg"].append(ang)
            errs["t"].append(np.linalg.norm(t_e - t_g))
            errs["s"].append(abs(a_e / a_g - 1))
            # end-to-end: warp view i's GT OBJECT into view j with each pose
            w_est = warp_sdf_affine(s_i["object"], a_e, R_e, t_e)
            w_gt = warp_sdf(s_i["object"], m_i, m_j)
            errs["iou_est"].append(sdf_agreement(s_j["object"], w_est)["iou"])
            errs["iou_gt"].append(sdf_agreement(s_j["object"], w_gt)["iou"])
            print(f"{inst} {i}->{j}: ang {ang:6.2f} deg | t {errs['t'][-1]*1000:5.1f} mgrid "
                  f"| s {errs['s'][-1]*100:4.1f}% | objIoU est {errs['iou_est'][-1]:.3f} "
                  f"vs gt {errs['iou_gt'][-1]:.3f} | cost {diag['cost0']:.4f}->{diag['cost']:.4f}")

    print("\n=== summary (n=%d) ===" % len(errs["deg"]))
    for k, scale, unit in [("deg", 1, "deg"), ("t", 1000, "milligrid"), ("s", 100, "%"),
                           ("iou_est", 1, ""), ("iou_gt", 1, "")]:
        v = np.array(errs[k]) * scale
        print(f"  {k:8s} mean {v.mean():7.3f} median {np.median(v):7.3f} "
              f"p90 {np.percentile(v, 90):7.3f} {unit}")


if __name__ == "__main__":
    main()
