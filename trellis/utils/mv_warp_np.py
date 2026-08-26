"""Numpy multi-view SDF warp helpers for dataset-side use (P4 recursive student).

Verbatim copies of the validated functions in tools/multiview_warp.py (P0,
EVAL_GUIDANCE.md §7.9: GT->GT IoU 0.94-0.97, band |sdf diff| ~1/10 voxel) —
duplicated here because trellis.datasets must not depend on tools/ path hacks.
KEEP IN SYNC with tools/multiview_warp.py.

Convention (found by composition sweep, NOT the naive reading):
x_view = s_aug * R_fixed^T @ x_canon + t_aug, voxel-CENTER sampling, SDF values
rescale by s_dst/s_src.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def sim_from_meta(meta):
    """Return (s, R, t, c0) with x_view = s * R @ (x_canon - c0) + c0 + t."""
    p = meta["pose"]
    return (float(p["s_aug"]), np.asarray(p["R_fixed"], dtype=np.float64),
            np.asarray(p["t_aug"], dtype=np.float64), np.asarray(p["c0"], dtype=np.float64))


def view_to_canon(pts, meta):
    s, R, t, c0 = sim_from_meta(meta)
    return (R @ ((pts - t) / s).T).T


def canon_to_view(pts, meta):
    s, R, t, c0 = sim_from_meta(meta)
    return (s * (R.T @ pts.T)).T + t


def grid_points(res=64, center=True):
    """World coords of grid samples, shape (res,res,res,3), i->x, j->y, k->z."""
    d = 2.0 / res
    ax = -1.0 + (np.arange(res) + (0.5 if center else 0.0)) * d
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def warp_sdf(sdf_src, meta_src, meta_dst, center=True, order=1, cval=1.0):
    """Resample sdf_src (on meta_src's grid) onto meta_dst's grid.

    For each dst voxel: dst grid -> canonical -> src grid, trilinear sample,
    then rescale values by s_dst/s_src so distances are in dst units.
    Outside the src grid, fill with cval (far-outside surrogate).
    """
    res = sdf_src.shape[0]
    d = 2.0 / res
    off = 0.5 if center else 0.0
    pts_dst = grid_points(res, center).reshape(-1, 3)
    pts_src = canon_to_view(view_to_canon(pts_dst, meta_dst), meta_src)
    idx = (pts_src + 1.0) / d - off                     # world -> fractional index
    out = map_coordinates(sdf_src, idx.T, order=order, mode="constant", cval=cval)
    s_src = sim_from_meta(meta_src)[0]
    s_dst = sim_from_meta(meta_dst)[0]
    return (out * (s_dst / s_src)).reshape(res, res, res)
