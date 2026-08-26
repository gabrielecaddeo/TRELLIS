"""Multi-view SDF warping for the fusion/consistency study (EVAL_GUIDANCE.md P0).

Each view f of a grasp instance stores hand/object SDFs on its own 64^3 grid over
[-1,1]^3; the view grid is a *similarity* transform of a canonical grasp frame
(meta['pose']: x_view = s_aug * R_fixed.T @ x_canon + t_aug  — determined empirically by
composition sweep: GT->GT warp IoU 0.958, band |sdf diff| 0.0035 ~ 1/9 voxel;
c0 is 0 in all inspected metas and R is stored as the view->canon rotation).
Because scales differ per view, warping SDF values between grids multiplies them
by the appropriate scale ratio (exact for true SDFs).

Conventions from meta['grid']: origin [-1,-1,-1], voxel 2/64, axes i->x,j->y,k->z,
negative inside. Sample positions are validated (voxel-center vs corner) by
`validate` below rather than assumed.

NOTE: sim_from_meta/view_to_canon/canon_to_view/grid_points/warp_sdf are
duplicated in trellis/utils/mv_warp_np.py for dataset-side use (P4 recursive
student) — keep the two copies in sync.
"""
import json
import os

import numpy as np
from scipy.ndimage import map_coordinates


def load_view(inst_dir, name, f):
    meta = json.load(open(os.path.join(inst_dir, f"{name}_f{f:03d}_meta.json")))
    sdfs = {}
    for part in ("hand", "object"):
        p = os.path.join(inst_dir, "sdfs", f"{name}_f{f:03d}__{part}.npy")
        if os.path.exists(p):
            sdfs[part] = np.load(p)
    return meta, sdfs


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


def sdf_agreement(a, b, band=0.1):
    """Metrics comparing two SDFs on the same grid: inside-IoU and mean |diff|
    over the near-surface band of `a` (|a| < band)."""
    ia, ib = a < 0, b < 0
    union = np.logical_or(ia, ib).sum()
    iou = np.logical_and(ia, ib).sum() / max(union, 1)
    m = np.abs(a) < band
    return {"iou": float(iou), "band_madiff": float(np.abs(a - b)[m].mean() if m.any() else np.nan),
            "inside_a": int(ia.sum()), "inside_b": int(ib.sum())}


def validate(inst_dir, name, pairs, center=True, part="object"):
    """Warp GT SDF of view i onto view j's grid and compare with view j's GT."""
    out = []
    for i, j in pairs:
        meta_i, sdf_i = load_view(inst_dir, name, i)
        meta_j, sdf_j = load_view(inst_dir, name, j)
        w = warp_sdf(sdf_i[part], meta_i, meta_j, center=center)
        m = sdf_agreement(sdf_j[part], w)
        m.update(i=i, j=j, part=part)
        out.append(m)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pairs", default="0-1,0-5,0-12,3-17,7-23")
    args = ap.parse_args()
    pairs = [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")]
    for center in (True, False):
        for part in ("object", "hand"):
            res = validate(args.inst_dir, args.name, pairs, center=center, part=part)
            ious = [r["iou"] for r in res]
            bd = [r["band_madiff"] for r in res]
            print(f"center={center} part={part}: IoU mean {np.mean(ious):.4f} min {np.min(ious):.4f} | band|diff| mean {np.mean(bd):.4f}")
            for r in res:
                print(f"   {r['i']:>2}->{r['j']:<2} IoU {r['iou']:.4f} band|diff| {r['band_madiff']:.4f} (in_gt {r['inside_b']}, in_warp {r['inside_a']})")


# ---------------------------------------------------------------------------
# Torch port for in-sampler use (P2 consistency guidance): precompute, per view,
# the grid_sample coordinates that pull that view's volume into the REF grid.
# Validated numerically against warp_sdf (scipy) by torch_warp_selftest below.
# ---------------------------------------------------------------------------

def torch_warp_coords(meta_src, meta_ref, res=64):
    """grid_sample coords (numpy [res,res,res,3], order z,y,x) mapping ref-grid
    voxel centers into meta_src's grid, plus the SDF value scale s_ref/s_src."""
    pts_ref = grid_points(res, center=True).reshape(-1, 3)
    pts_src = canon_to_view(view_to_canon(pts_ref, meta_ref), meta_src)
    # volume stored [x,y,z] as dims (D,H,W) => grid last dim must be (W=z,H=y,D=x)
    coords = pts_src[:, ::-1].reshape(res, res, res, 3)
    s_src = sim_from_meta(meta_src)[0]
    s_ref = sim_from_meta(meta_ref)[0]
    return coords, s_ref / s_src


def torch_warp(vol, coords, scale, pad_value=1.0):
    """vol [B,1,64,64,64] (dims x,y,z), coords torch [64,64,64,3] on same device.
    Returns the volume resampled onto the ref grid, values scaled, out-of-range
    filled with pad_value. Differentiable w.r.t. vol."""
    import torch
    import torch.nn.functional as tF
    B = vol.shape[0]
    g = coords[None].expand(B, -1, -1, -1, -1)
    # pad with (pad_value/scale) so padded regions end up at pad_value after scaling
    shifted = vol - pad_value / scale
    out = tF.grid_sample(shifted, g, mode="bilinear",
                         padding_mode="zeros", align_corners=False)
    return (out + pad_value / scale) * scale


def torch_warp_selftest(inst_dir, name, i, j, device="cpu", atol=0.02):
    """Compare torch_warp vs warp_sdf on GT; returns (max_abs_diff, ok)."""
    import torch
    meta_i, sdf_i = load_view(inst_dir, name, i)
    meta_j, _ = load_view(inst_dir, name, j)
    ref = warp_sdf(sdf_i["object"], meta_i, meta_j)
    coords, scale = torch_warp_coords(meta_i, meta_j)
    v = torch.from_numpy(sdf_i["object"].copy()).float()[None, None].to(device)
    c = torch.from_numpy(np.ascontiguousarray(coords)).float().to(device)
    out = torch_warp(v, c, scale).cpu().numpy()[0, 0]
    interior = np.abs(ref) < 0.9  # skip far-field pad differences
    d = float(np.abs(out - ref)[interior].max())
    return d, d < atol
