"""DexYCB calibration -> GT pose blocks for dex-full instances (EVAL_GUIDANCE.md §7.27).

Chain (determined empirically 2026-09-09 by composition sweep against the hand
registration: GT->GT object IoU 0.986-0.988, rotation agreement 0.1-0.4 deg):
  x_grid = s_norm * x_render + t_norm            (per-instance fit, R = I: the
                                                  export voxelizes in the render frame)
  x_cam   = F @ x_render,  F = diag(1,-1,-1)     (pyrender convention, renders_cond/meta.json)
  x_world = R_c @ x_cam + t_c                    (calibration/extrinsics_<date>/extrinsics.yml,
                                                  3x4 row-major [R|t], camera->world,
                                                  master 840412060917 = identity)
Sequence date -> the latest extrinsics set dated <= it.
Meta convention of tools/multiview_warp.py: x_view = s_aug * R_fixed.T @ x_canon + t_aug
with canonical := world (meters)  =>  s_aug = s_norm, R_fixed = R_c @ F,
t_aug = t_norm - s_norm * F @ R_c.T @ t_c.
"""
import os, glob, json
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

F = np.diag([1.0, -1.0, -1.0])
DX = 2.0 / 64
_ext_cache = {}


def ext_dates(root):
    return sorted(os.path.basename(p).split("_")[1] for p in glob.glob(os.path.join(root, "calibration", "extrinsics_*")))


def extrinsics_for(root, seq_date):
    d = [e for e in ext_dates(root) if e <= seq_date][-1]
    if d not in _ext_cache:
        f = glob.glob(os.path.join(root, "calibration", f"extrinsics_{d}_*", "extrinsics.yml"))[0]
        y = yaml.load(open(f), Loader=yaml.UnsafeLoader)
        _ext_cache[d] = {k: np.array(v, dtype=np.float64).reshape(3, 4) for k, v in y["extrinsics"].items()}
    return d, _ext_cache[d]


def obj_vertices(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])


def _sample(sdf, pts):
    return map_coordinates(sdf, ((pts + 1.0) / DX - 0.5).T, order=1, mode="nearest")


def fit_norm(root, inst):
    """x_grid = s * x_render + t: least squares of the hand+object mesh vertices onto
    the zero level of their SDFs (bbox init from the interior-voxel indices)."""
    V, S = {}, {}
    for p in ("hand", "object"):
        V[p] = obj_vertices(os.path.join(root, "renders_cond", inst, f"{p}.obj"))
        S[p] = np.load(os.path.join(root, "data_pose_norm", inst, "sdfs", f"{inst}_f000__{p}.npy")).astype(np.float32)
    vm = np.concatenate(list(V.values()))
    vg = np.concatenate([np.load(os.path.join(root, "data_pose_norm", inst, "idxs", f"{inst}_f000__{p}.npy"))
                         for p in ("hand", "object")])
    vg = -1.0 + (vg + 0.5) * DX
    s0 = float(((vg.max(0) - vg.min(0) + DX) / (vm.max(0) - vm.min(0))).mean())
    t0 = (vg.min(0) - DX / 2) - s0 * vm.min(0)

    def resid(p):
        return np.concatenate([np.clip(_sample(S[k], p[0] * V[k] + p[1:]), -0.1, 0.1) for k in V])
    sol = least_squares(resid, np.r_[s0, t0], loss="soft_l1", f_scale=0.02)
    return float(sol.x[0]), sol.x[1:].copy(), float(np.abs(sol.fun).mean())


def gt_pose_block(root, inst, camera, seq_date):
    d, ext = extrinsics_for(root, seq_date)
    Rc, tc = ext[camera][:, :3], ext[camera][:, 3]
    s, t, res = fit_norm(root, inst)
    R_fixed = Rc @ F
    t_aug = t - s * (F @ Rc.T @ tc)
    return {"R_fixed": R_fixed.tolist(), "s_aug": s, "t_aug": t_aug.tolist(), "c0": [0.0, 0.0, 0.0],
            "about": "similarity about shared pivot c0; canonical = DexYCB world frame (meters) from "
                     f"calibration/extrinsics_{d} (camera {camera}); x_view = s_aug*R_fixed.T@x_canon + t_aug",
            "fit_residual": res, "extrinsics_set": d}
