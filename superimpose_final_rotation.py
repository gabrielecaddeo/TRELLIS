import os
import trimesh
import numpy as np
import pyvista as pv
import mesh2sdf
import utils3d
import glob
import argparse
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from scipy.ndimage import map_coordinates, binary_fill_holes
import igl
import traceback
from itertools import permutations
import sys
import json
import skfmm
import timeit
from PIL import Image
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from collections import deque

# ---------- helpers (unit2 = [-1,1]) ----------
def grid_centers_u2(res):
    xs = (np.arange(res, dtype=np.float64) + 0.5) * (2.0/res) - 1.0
    X,Y,Z = np.meshgrid(xs,xs,xs, indexing='ij')
    return np.stack([X,Y,Z], axis=-1)    # (res,res,res,3)

def centers_to_float_indices_u2(centers, res):
    # [-1,1] centers -> continuous indices so that index i corresponds to center i+0.5
    return (centers + 1.0) * (res/2.0) - 0.5

def sample_ST_no_clip_u2(V_u2, R, margin_vox=1, res=64, scale_range=(0.5,1.0), rng=None, slack=1e-4):
    if rng is None:
        rng = np.random.default_rng()
    m = (2.0/res) * margin_vox  # voxel margin in unit2

    vmin = V_u2.min(axis=0); vmax = V_u2.max(axis=0)
    c0   = 0.5*(vmin+vmax)
    Vr   = (V_u2 - c0) @ R
    vminR = Vr.min(axis=0); vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)

    # fit inside [-1,1] => half-box=1 → (1-m-slack)
    s_max = np.min((2.0 - 2.0*m - 2.0*slack) / widths)
    s_hi  = min(scale_range[1], s_max)
    s_lo  = min(scale_range[0], s_hi)
    s     = rng.uniform(s_lo, s_hi)
    s     = min(s, 0.999*s_max)

    t_lo = -1.0 + m + slack - s*vminR
    t_hi =  1.0 - m - slack - s*vmaxR
    t    = 0.5*(t_lo + t_hi)

    V_pose = ((V_u2 - c0) @ R) * s + t
    return V_pose, s, t, c0

def sample_ST_no_clip_u2_randT(
    V_u2, R, *,
    margin_vox=1, res=64,
    scale_range=(0.7, 1.0),
    rng=None, slack=1e-4,
    extra_inner_margin_vox=0,   # push object further from the walls (in voxels)
    center_prob=0.0             # >0 to sometimes center (e.g., 0.2)
):
    """
    Same as your sample_ST_no_clip_u2 but picks a *random* t in [t_lo, t_hi].
    Works in unit2 domain [-1,1]^3. Row-vector convention: X' = ((X-c0) @ R)*s + t
    """
    if rng is None:
        rng = np.random.default_rng()

    # margins in unit2
    m = float(margin_vox)/res + float(extra_inner_margin_vox)/res

    # AABB center and rotate
    vmin = V_u2.min(axis=0); vmax = V_u2.max(axis=0)
    c0   = 0.5*(vmin+vmax)
    Vr   = (V_u2 - c0) @ R
    vminR = Vr.min(axis=0); vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)

    # Max allowable scale so the rotated box fits inside [-1,1]^3 with margin m
    s_max = np.min((2.0 - 2.0*m - 2.0*slack) / widths)   # box side is 2 in unit2
    if not np.isfinite(s_max) or s_max <= 0:
        raise RuntimeError("No feasible scale with given rotation/margin.")

    s_hi = min(scale_range[1], s_max)
    s_lo = min(scale_range[0], s_hi)
    s    = rng.uniform(s_lo, s_hi)
    s    = min(s, 0.999*s_max)  # stay comfortably inside

    # Feasible translation interval per axis
    t_lo = -1.0 + m + slack - s*vminR
    t_hi =  1.0 - m - slack - s*vmaxR
    if np.any(t_lo > t_hi):
        raise RuntimeError("Empty translation range; reduce scale/margins.")

    # Random translation (optionally sometimes take the midpoint)
    if rng.random() < center_prob:
        t = 0.5*(t_lo + t_hi)
    else:
        t = rng.uniform(t_lo, t_hi)

    # Apply
    V_pose = ((V_u2 - c0) @ R) * s + t
    return V_pose, s, t, c0

def occ_from_V_pose_u2(V_pose_u2, F, res=64):
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(V_pose_u2.astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

        # unit2 cube: [-1,1]^3, voxel size = 2/res
        vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            m, voxel_size=2.0/res,
            min_bound=(-1.0, -1.0, -1.0), max_bound=(1.0, 1.0, 1.0)
        )
        occ = np.zeros((res,res,res), dtype=bool)
        if len(vg.get_voxels()):
            ijk = np.array([v.grid_index for v in vg.get_voxels()], dtype=int)
            occ[ijk[:,0], ijk[:,1], ijk[:,2]] = True
        return occ

def normalize_to_unit2(V_raw, eps=1e-6):
    # center at AABB center and scale uniformly so the largest dimension fits in [-1,1]
    vmin = V_raw.min(axis=0)
    vmax = V_raw.max(axis=0)
    c    = 0.5 * (vmin + vmax)
    ext  = (vmax - vmin).max()  # largest side of the AABB
    if ext <= 0:
        raise ValueError("Degenerate mesh: zero extent")
    s = (2.0 - 2.0*eps) / ext   # fit *inside* [-1,1] with a tiny margin
    V_u2 = (V_raw - c) * s
    return V_u2, s, c

def canonical_sdf_u2_norm(mesh_path, res0=128, do_normalize=True):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    V_raw = m.vertices.view(np.ndarray).astype(np.float64)
    F     = m.faces.view(np.ndarray).astype(np.int32)
    # print("Pre-normalization AABB:", V_raw.min(0), V_raw.max(0))
    if do_normalize:
        V_u2, s_norm, c_norm = normalize_to_unit2(V_raw)  # now truly in [-1,1]^3
    else:
        # Only use this if you KNOW your input is already in unit2
        V_u2 = np.clip(V_raw, -1 + 1e-6, 1 - 1e-6)
        s_norm, c_norm = 1.0, np.zeros(3)
    # print("Post-normalization AABB:", V_u2.min(0), V_u2.max(0))
    level = 2.0/res0
    sdf0_raw, _ = mesh2sdf.compute(V_u2, F, res0, fix=False, level=level, return_mesh=True)
    sdf0 = np.asarray(sdf0_raw).reshape(res0, res0, res0)

    # return normalization so you can record provenance if needed
    return V_u2, F, sdf0, s_norm, c_norm

# ---------- step 1: canonical high-res SDF ----------
def canonical_sdf_u2(mesh_path, res0=128):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    V = m.vertices.view(np.ndarray).astype(np.float64)
    F = m.faces.view(np.ndarray).astype(np.int32)
    # clamp to unit2 if you already normalized upstream; otherwise normalize first
    V = np.clip(V, -1 + 1e-6, 1 - 1e-6)

    level = 2.0/res0
    sdf0_raw, _ = mesh2sdf.compute(V, F, res0, fix=False, level=level, return_mesh=True)
    sdf0 = np.asarray(sdf0_raw).reshape(res0,res0,res0)
    return V, F, sdf0

# ---------- step 2: resample canonical SDF into posed 64³ ----------
def resample_sdf_u2(sdf0, R, s, t, c0, target_res=64, order=3, cval=4.0):
    res0 = sdf0.shape[0]
    Y = grid_centers_u2(target_res)                             # (R,R,R,3)
    X = ((Y - t.reshape(1,1,1,3))/float(s)) @ R.T + c0          # inverse map
    IJK = centers_to_float_indices_u2(X, res0)
    i,j,k = IJK[...,0], IJK[...,1], IJK[...,2]
    phi = map_coordinates(sdf0, [i,j,k], order=order, mode='constant', cval=cval)
    if s != 1.0: phi = phi * float(s)                           # scale distances
    return phi.reshape(target_res,target_res,target_res), Y

# ---------- step 3: pose geometry + sign from posed mesh ----------
def sign_from_winding_u2(V_pose_u2, F, Y):
    P = Y.reshape(-1,3)
    w = igl.fast_winding_number(V_pose_u2, F, P)    # (R^3,)
    inside = (w > 0.5).reshape(Y.shape[:3])
    return inside

# ---------- step 4: narrow-band refinement with exact distances ----------
def refine_band_exact_u2(phi, inside, V_pose_u2, F, Y, band_vox=2, dx=2.0/64):
    band = np.abs(phi) < (band_vox*dx)
    if not np.any(band):
        # fall back to sign fix only
        return np.where(inside, -np.abs(phi), np.abs(phi))
    P = Y[band].reshape(-1,3)
    d2, _, _ = igl.point_mesh_squared_distance(P, V_pose_u2, F)
    d = np.sqrt(d2)
    phi_ref = phi.copy()
    signs = np.where(inside[band], -1.0, +1.0)
    phi_ref[band] = signs * d
    # ensure outside region keeps the original magnitude
    phi_ref[~band] = np.where(inside[~band], -np.abs(phi[~band]), np.abs(phi[~band]))
    return phi_ref

def collage_from_grid_bbox_preserve_aspect_negz(
    image_rgba: Image.Image,
    occ: np.ndarray,          # bool [R,R,R], posed occupancy (grid index order: [x,y,z])
    res: int = 64,
    out_size: int = 1024,
    x_right: bool = True,     # keep True: image x to the right
    y_up: bool = True         # << set True to make +y go UP in image (translation only)
) -> Image.Image:
    assert image_rgba.mode == "RGBA"
    R = res
    H = W = int(out_size)
    px_per_vox = float(W) / float(R)

    # 1) project along z to get footprint on x–y
    mask_xy = occ.any(axis=2)     # (x,y)
    img_mask = mask_xy.T          # (rows=y, cols=x) → image order

    # 2) apply axis sign conventions to the MASK ONLY (no sprite flip)
    if not x_right:
        img_mask = img_mask[:, ::-1]   # mirror columns if you ever need left-positive
    if y_up:
        img_mask = img_mask[::-1, :]   # invert y sign: +y goes upward in image

    if not img_mask.any():
        return Image.new("RGBA", (W, H), (0,0,0,0))

    # 3) bbox of the footprint in image coords
    rows, cols = np.where(img_mask)
    ymin, ymax = rows.min(), rows.max()
    xmin, xmax = cols.min(), cols.max()
    w_vox = xmax - xmin + 1
    h_vox = ymax - ymin + 1

    xmin_px = int(round(xmin * px_per_vox))
    ymin_px = int(round(ymin * px_per_vox))
    w_px_box = max(1, int(round(w_vox * px_per_vox)))
    h_px_box = max(1, int(round(h_vox * px_per_vox)))

    # 4) crop sprite from alpha
    A = np.array(image_rgba.getchannel("A"))
    ys, xs = np.nonzero(A > 0)
    if len(xs) == 0:
        return Image.new("RGBA", (W, H), (0,0,0,0))
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sprite = image_rgba.crop((x0, y0, x1+1, y1+1))
    sw, sh = sprite.size

    # 5) uniform scale to fit inside the voxel box (preserve aspect)
    scale = max(1e-6, min(w_px_box / sw, h_px_box / sh))
    new_w = max(1, int(round(sw * scale)))
    new_h = max(1, int(round(sh * scale)))
    sprite_resized = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6) center inside the target box
    dx_px = (w_px_box - new_w) // 2
    dy_px = (h_px_box - new_h) // 2
    paste_x = xmin_px + dx_px
    paste_y = ymin_px + dy_px

    # 7) paste onto canvas (no sprite flip)
    canvas = Image.new("RGBA", (W, H), (0,0,0,0))
    paste_x = max(-new_w, min(W, paste_x))
    paste_y = max(-new_h, min(H, paste_y))
    canvas.alpha_composite(sprite_resized, dest=(paste_x, paste_y))
    return canvas


def collage_from_meta_bbox_preserve_aspect(
    image_rgba: Image.Image,
    pose2d_meta: dict,
    out_size: int = 1024,
):
    """
    Place the RGBA sprite into a square canvas so its size+translation in the
    image plane matches the saved grid bbox (no stretching; uniform scale).
    """
    assert image_rgba.mode == "RGBA"
    W = H = int(out_size)
    res      = int(pose2d_meta["res"])
    bbox_xy  = pose2d_meta["bbox_xy"]
    x_right  = bool(pose2d_meta.get("x_right", True))
    y_up     = bool(pose2d_meta.get("y_up", True))

    canvas = Image.new("RGBA", (W, H), (0,0,0,0))
    if bbox_xy is None:
        return canvas  # empty mask case

    xmin, ymin, xmax, ymax = bbox_xy
    # convert voxel bbox → pixel bbox
    px_per_vox = float(W) / float(res)
    xmin_px = int(round(xmin * px_per_vox))
    ymin_px = int(round(ymin * px_per_vox))
    w_px_box = max(1, int(round((xmax - xmin + 1) * px_per_vox)))
    h_px_box = max(1, int(round((ymax - ymin + 1) * px_per_vox)))

    # crop sprite from Blender alpha (original render content)
    A = np.asarray(image_rgba.getchannel("A"))
    ys, xs = np.nonzero(A > 0)
    if len(xs) == 0:
        return canvas
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sprite = image_rgba.crop((x0, y0, x1+1, y1+1))
    sw, sh = sprite.size

    # uniform scale to fit inside target box (preserve aspect)
    scale = max(1e-6, min(w_px_box / sw, h_px_box / sh))
    new_w = max(1, int(round(sw * scale)))
    new_h = max(1, int(round(sh * scale)))
    sprite_resized = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # center within the target box
    dx_px = (w_px_box - new_w) // 2
    dy_px = (h_px_box - new_h) // 2
    paste_x = xmin_px + dx_px
    paste_y = ymin_px + dy_px

    # If you ever need to invert translation conventions post-hoc,
    # you can modify (paste_x, paste_y) here using x_right/y_up flags.
    # (We already encoded those when producing bbox_xy, so no flips now.)

    paste_x = max(-new_w, min(W, paste_x))
    paste_y = max(-new_h, min(H, paste_y))
    canvas.alpha_composite(sprite_resized, dest=(paste_x, paste_y))
    return canvas

def sample_ST_no_clip_u2_xyOnly(
    V_u2, R, *,
    margin_vox=1, res=64,
    scale_range=(0.6, 1.0),
    rng=None, slack=1e-4,
    extra_inner_margin_vox=0.0,  # optional extra gap from walls (voxels)
    center_z=True                # keep z centered (midpoint). If False, z=0.
):
    """
    Pose sampler in unit2 ([-1,1]^3) with:
      - random scale 's' (within feasibility)
      - random in-plane translation (t_x, t_y)
      - NO random depth translation (t_z); we either center it or set to 0.
    Row-vector convention: X' = ((X - c0) @ R) * s + t
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    # unit2 margin in world units
    m = float(margin_vox)/res + float(extra_inner_margin_vox)/res

    # rotate around AABB center so widths are wrt axes
    vmin = V_u2.min(axis=0); vmax = V_u2.max(axis=0)
    c0   = 0.5*(vmin + vmax)
    Vr   = (V_u2 - c0) @ R
    vminR = Vr.min(axis=0); vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)   # extents along x,y,z after rotation

    # feasible max scale so box fits inside [-1,1]^3 with margin m
    s_max = np.min((2.0 - 2.0*m - 2.0*slack) / widths)
    if not np.isfinite(s_max) or s_max <= 0:
        raise RuntimeError("No feasible scale; reduce margin/rotation or rescale mesh.")

    s_hi = min(scale_range[1], s_max)
    s_lo = min(scale_range[0], s_hi)
    s    = rng.uniform(s_lo, s_hi)
    s    = min(s, 0.999*s_max)  # keep a bit inside

    # feasible translation ranges per axis
    t_lo = -1.0 + m + slack - s*vminR
    t_hi =  1.0 - m - slack - s*vmaxR
    if np.any(t_lo > t_hi):
        raise RuntimeError("Empty translation range even after scaling; tighten margins.")

    # sample in-plane translation (x,y) uniformly in feasible interval
    tx = rng.uniform(t_lo[0], t_hi[0])
    ty = rng.uniform(t_lo[1], t_hi[1])

    # depth translation (z): keep centered or fixed to 0 (no randomization)
    if center_z:
        tz = 0.5*(t_lo[2] + t_hi[2])
    else:
        tz = 0.0
        # if you prefer: clamp tz into [t_lo[2], t_hi[2]] to guarantee containment
        tz = np.clip(tz, t_lo[2], t_hi[2])

    t = np.array([tx, ty, tz], dtype=np.float64)

    # apply to vertices if you want the posed mesh back
    V_pose = ((V_u2 - c0) @ R) * s + t
    return V_pose, s, t, c0

def pose2d_meta_from_occ(
    occ: np.ndarray,             # bool [R,R,R] in posed frame
    res: int = 64,
    view_axis: str = '-z',       # you use camera along -Z
    x_right: bool = True,        # image x to the right
    y_up: bool = True            # image y is upward (translation-only flip)
):
    assert occ.shape == (res,res,res)
    axis = view_axis.strip().lower()
    assert axis[-1] in ('x','y','z')
    # project to XY plane (X→columns, Y→rows)
    if axis[-1] == 'z':
        mask_xy = occ.any(axis=2)      # (x,y)
    elif axis[-1] == 'y':
        mask_xy = occ.any(axis=1)
    else:
        mask_xy = occ.any(axis=0)

    img_mask = mask_xy.T               # (rows=y, cols=x)
    if not x_right:
        img_mask = img_mask[:, ::-1]
    if y_up:
        img_mask = img_mask[::-1, :]

    if not img_mask.any():
        # degenerate; encode "empty"
        return {
            "res": res, "view_axis": view_axis,
            "x_right": x_right, "y_up": y_up,
            "bbox_xy": None
        }

    rows, cols = np.where(img_mask)
    ymin, ymax = int(rows.min()), int(rows.max())
    xmin, xmax = int(cols.min()), int(cols.max())

    return {
        "res": res,
        "view_axis": view_axis,
        "x_right": x_right,
        "y_up": y_up,
        # bbox in **grid index coordinates** of the image plane
        "bbox_xy": [xmin, ymin, xmax, ymax]
    }


def save_voxelization_and_sdf(
    out_dir, name, occ, sdf, res=64,
    R_fixed=None, s_aug=None, t_aug=None, c0=None,
    s_norm=None, c_norm=None,
    frame_id=None, margin_vox=1,
    space="unit2"  # "unitcube" -> [-0.5,0.5]^3 ; "unit2" -> [-1,1]^3
):
    """
    Saves posed, ready-to-use data (no need to reapply transforms):

      voxels/{name}.ply    : voxel centers as points (in grid coords)
      occs/{name}.npy      : bool [res,res,res]
      sdfs/{name}.npy      : float32 [res,res,res]
      idxs/{name}.npy      : int32 [N,3] (occupied indices)
      {name}_meta.json     : grid + pose metadata (R, s, t, c0, conventions)
    """

    # --- folders ---
    #os.makedirs(os.path.join(out_dir, "voxels"), exist_ok=True)
    #os.makedirs(os.path.join(out_dir, "occs"),   exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sdfs"),   exist_ok=True)
    os.makedirs(os.path.join(out_dir, "idxs"),   exist_ok=True)

    # --- basic checks ---
    assert occ.shape == (res, res, res)
    assert sdf.shape == (res, res, res)
    occ = occ.astype(np.bool_)
    sdf = sdf.astype(np.float32)

    # --- grid convention ---
    if space == "unitcube":
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        voxel_size = 1.0 / res
    elif space == "unit2":
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        voxel_size = 2.0 / res
    else:
        raise ValueError(f"Unknown space '{space}'")

    # --- indices & centers in the SAME grid space as sdf/occ ---
    idx = np.argwhere(occ)  # (N,3) with i,j,k
    centers = origin + (idx.astype(np.float64) + 0.5) * voxel_size

    # --- write files ---
    # PLY of centers (already in posed grid coordinates)
    # expects utils3d.io.write_ply(path, Nx3 float32 array)
    #utils3d.io.write_ply(os.path.join(out_dir, "voxels", f"{name}.ply"), centers.astype(np.float32))

    #np.save(os.path.join(out_dir, "occs", f"{name}.npy"), occ)
    np.save(os.path.join(out_dir, "sdfs", f"{name}.npy"), sdf)
    np.save(os.path.join(out_dir, "idxs", f"{name}.npy"), idx.astype(np.int32))

    image_info = pose2d_meta_from_occ(occ, res=res)

    # --- metadata (everything needed to interpret the files later) ---
    meta = {
        "grid": {
            "res": int(res),
            "space": space,                              # "unitcube" or "unit2"
            "origin": origin.tolist(),                   # 3-vector
            "voxel_size": float(voxel_size),
            "bbox_min": origin.tolist(),
            "bbox_max": (origin + voxel_size * res).tolist(),
            "index_axes": "i->x, j->y, k->z",            # axis convention
            "indexing": "np.meshgrid(..., indexing='ij')",
            "sdf_negative_inside": True,
            "iso_value": 0.0
        },
        "pose": {
            "R_fixed": None if R_fixed is None else np.asarray(R_fixed, dtype=float).tolist(),
            "s_aug": None if s_aug is None else float(s_aug),
            "t_aug": None if t_aug is None else np.asarray(t_aug, dtype=float).tolist(),
            "c0":    None if c0    is None else np.asarray(c0,    dtype=float).tolist(),
            "about": "similarity about AABB center c0"
        },
        "provenance": {
            "frame_id": None if frame_id is None else int(frame_id),
            "margin_vox": int(margin_vox),
            "pipeline": "mesh2sdf->resample(+/-)->(optional)skfmm",
        },
        "stats": {
            "occ_voxels": int(idx.shape[0]),
            "sdf_min": float(sdf.min()),
            "sdf_max": float(sdf.max()),
            "sdf_mean": float(sdf.mean()),
        },
        "canonical": {
            "do_normalize": True,
            "s_norm": float(s_norm),
            "c_norm": np.asarray(c_norm, dtype=float).tolist(),
            "about": "x_norm = (x_raw - c_norm) * s_norm"
        },
        "pose2d_meta": image_info
    }
    with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "occ":   os.path.join(out_dir, "occs", f"{name}.npy"),
        "sdf":   os.path.join(out_dir, "sdfs", f"{name}.npy"),
        "idx":   os.path.join(out_dir, "idxs", f"{name}.npy"),
        "ply":   os.path.join(out_dir, "voxels", f"{name}.ply"),
        "meta":  os.path.join(out_dir, f"{name}_meta.json"),
    }


def post_adjustment_sdf_mod(instance_name, root, visualize=True, save_bool=False, save_folder = None):
    print(f"Processing instance: {instance_name}")

    # Now, construct the path to the corresponding mesh file.
    ply_mesh_path = os.path.join(root, 'renders', instance_name, "mesh.ply")

    if not os.path.exists(ply_mesh_path):
        print(f"  [Warning] Corresponding mesh file not found, skipping: {ply_mesh_path}")
        return

    print(f"  Mesh path: {ply_mesh_path}")
    if save_bool:
        # Ensure the directory exists
        if not os.path.isdir(os.path.join(save_folder, instance_name)):
        # Save the SDF to the specified path
            os.makedirs(os.path.join(save_folder, instance_name), exist_ok=True)

        if not os.path.isdir(os.path.join(save_folder, instance_name, "sdfs")):
            os.makedirs(os.path.join(save_folder, instance_name, "sdfs"), exist_ok=True)
        if not os.path.isdir(os.path.join(save_folder, instance_name, "voxels")):
            os.makedirs(os.path.join(save_folder, instance_name, "voxels"), exist_ok=True)
        if not os.path.isdir(os.path.join(save_folder, instance_name, "idxs")):
            os.makedirs(os.path.join(save_folder, instance_name, "idxs"), exist_ok=True)
        sdf_folder = os.path.join(save_folder, instance_name, "sdfs")
    list_jsons = sorted(glob.glob(os.path.join(save_folder, instance_name, '*.json')))
    # print(len(list_jsons))
    #final_val=24
    if len(list_jsons)==24:
        #print(f"  SDFs already exist for all frames, skipping: {instance_name}")
        return
    meta   = json.load(open(os.path.join(root, 'renders_cond', instance_name, 'transforms.json')))
    for frame_id in range(24):
        try:

            # start_time = timeit.default_timer()
            # V_u2, F, sdf0 = canonical_sdf_u2(ply_mesh_path, res0=64)   # 128 or 256 recommended
            V_u2, F, sdf0, s_norm, c_norm = canonical_sdf_u2_norm(ply_mesh_path, res0=64)

            R_fixed = np.array(meta["frames"][frame_id]["transform_matrix"], dtype=float)[:3,:3]
            V_pose_u2, s_aug, t_aug, c0 = sample_ST_no_clip_u2_xyOnly(V_u2, R_fixed, margin_vox=1, res=64, scale_range=(0.5,1.0))

            # 2) Resample canonical SDF into posed 64³
            phi, Y = resample_sdf_u2(sdf0, R_fixed, s_aug, t_aug, c0, target_res=64, order=3, cval=4.0)

            # 3) Get sign from posed mesh (robust)
            inside = sign_from_winding_u2(V_pose_u2, F, Y)
            #print(timeit.default_timer() - start_time)
            # 4) Narrow-band refinement (keeps thin legs)
            sdf = refine_band_exact_u2(phi, inside, V_pose_u2, F, Y, band_vox=2, dx=2.0/64)-0.012
            #print(timeit.default_timer() - start_time)
            dx = 2.0/64
            sdf = skfmm.distance(np.ma.MaskedArray(sdf, mask=np.zeros_like(sdf, dtype=bool)), dx=dx)
            # 5) (Optional) occupancy for visualization, from posed mesh in unit2
            ss = occ_from_V_pose_u2(V_pose_u2, F, res=64)

            # ss = binary_fill_holes(ss)

            if save_bool:
                _ = save_voxelization_and_sdf(
                    out_dir    = os.path.join(save_folder, instance_name),
                    name       = f"{instance_name}_f{frame_id:03d}",
                    occ        = ss,                               # bool [64,64,64]
                    sdf        = sdf,                               # float32 [64,64,64]
                    res        = 64,
                    R_fixed    = R_fixed,                           # 3x3 rotation used for pose
                    s_aug      = s_aug,                             # sampled scale
                    t_aug      = t_aug,                             # sampled translation
                    c0         = c0,                                # center used for about-center transform
                    s_norm= s_norm,                           # canonical normalization scale
                    c_norm= c_norm,                           # canonical normalization center
                    frame_id   = frame_id,
                    margin_vox = 1,
                )

                print(f"Saved {instance_name}_f{frame_id:03d}", )
            if visualize:

                if save_bool:
                    print(f"  SDF loaded from: {sdf_folder}")
                    sdf= np.load(os.path.join(sdf_folder, f"{instance_name}_f{frame_id:03d}.npy"))
                    ss = np.load(os.path.join(save_folder, instance_name, "occs", f"{instance_name}_f{frame_id:03d}.npy"))
                # sdf = shift(sdf_tensor, shift_in_grid.numpy())
                # === Data preparation for plotting ===
                threshold = 0.005

                # --- Prepare Voxel Shell Surface ---
                voxel_grid_np = ss.astype(np.uint8)
                grid = pv.ImageData()
                grid.dimensions = voxel_grid_np.shape
                grid.point_data['voxels'] = voxel_grid_np.flatten(order='F')
                voxel_shell_surface = grid.contour([0.5], scalars='voxels')

                # --- Prepare SDF Grid for Contouring (used by plot 2) ---
                sdf_pyvista_grid = pv.ImageData()
                sdf_pyvista_grid.dimensions = sdf.shape
                sdf_pyvista_grid.point_data['sdf'] = sdf.flatten(order='F')

                # === Static Plot (for reference) ===
                print("\n--- Displaying Static Plot ---")
                static_plotter = pv.Plotter()
                static_plotter.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.5)

                sdf_near_zero_mask = np.abs(sdf) <= threshold
                sdf_coords = np.argwhere(sdf_near_zero_mask)
                sdf_point_cloud = pv.PolyData(sdf_coords)

                static_plotter.add_points(sdf_point_cloud, color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True, label=f"SDF |x| <= {threshold}")
                static_plotter.add_axes()
                static_plotter.show_grid()
                static_plotter.add_legend()
                static_plotter.show(title=f"Static SDF vs Voxel Shell for {instance_name}", window_size=[1000, 800])

                # === PLOT 1: Interactive Volumetric |SDF| Region ===
                print("\n--- Displaying Interactive Plot 1: Volumetric |SDF| Region ---")
                p1 = pv.Plotter(window_size=[1000, 800])
                p1.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
                p1.add_text("Voxel Shell + Interactive |SDF| Region", font_size=12)

                def callback_abs_sdf_volume(x_value):
                    p1.remove_actor("sdf_point_cloud", render=False)
                    dynamic_threshold = x_value + threshold

                    if dynamic_threshold >= 0:
                        mask = np.abs(sdf) <= dynamic_threshold
                        coords = np.argwhere(mask)
                        if coords.size > 0:
                            point_cloud = pv.PolyData(coords)
                            p1.add_points(point_cloud, name="sdf_point_cloud", color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True)

                p1.add_slider_widget(callback=callback_abs_sdf_volume, rng=[-0.5, 0.5], value=0.0, title="x, where |sdf| <= x + threshold", style='modern')
                p1.add_axes()
                p1.show_grid()
                p1.show(title=f"Interactive Volumetric |SDF| for {instance_name}")

                # === PLOT 2: Interactive SDF Level Set ===
                print("\n--- Displaying Interactive Plot 2: SDF Level Set ---")
                p2 = pv.Plotter(window_size=[1000, 800])
                p2.add_mesh(voxel_shell_surface, color="green", label="Voxel Shell", opacity=0.3)
                p2.add_text("Voxel Shell + Interactive SDF Level", font_size=12)

                def callback_single_sdf(threshold2):
                    p2.remove_actor("sdf_iso_single", render=False)
                    iso = sdf_pyvista_grid.contour([threshold2], scalars='sdf')
                    p2.add_mesh(iso, name="sdf_iso_single", color="purple")

                p2.add_slider_widget(callback=callback_single_sdf, rng=[-1.0, 1.0], value=0.0, title="threshold2, where sdf <= threshold2", style='modern')
                p2.add_axes()
                p2.show_grid()
                p2.show(title=f"Interactive SDF for {instance_name}")

            # === PLOT 3: Interactive SDF Slice with Binned Contours ===
                print("\n--- Displaying Plot 3: Interactive SDF Slice with Binned Contours ---")
                p3 = pv.Plotter(window_size=[1000, 800])
                p3.set_background('darkgrey') # Set background to see empty slices clearly

                # --- Generate Coherent Logarithmic Contour Levels ---
                sdf_min, sdf_max = sdf.min(), sdf.max()
                num_bins = 10

                # To create a log scale over a range that includes negative numbers,
                # we map the SDF range to a positive interval, log-space it, then map back.
                delta = sdf_max - sdf_min
                # Start logspace from a small fraction of the total range to avoid log(0)
                epsilon = 1e-4 * delta
                log_spaced_increments = np.logspace(np.log10(epsilon), np.log10(delta + epsilon), num_bins + 1)
                # Shift these increments to start from sdf_min
                contour_levels = sdf_min + log_spaced_increments - epsilon
                contour_levels[0] = sdf_min # Ensure the first level is exactly the minimum
                contour_levels[-1] = sdf_max # Ensure the last level is exactly the maximum

                # Define a callback function for the slider
                def update_slice_y(y_pos):
                    # Remove previous actors to prevent overlap
                    p3.remove_actor("slice_mesh", render=False)
                    p3.remove_actor("slice_edges", render=False)

                    # Create a 2D slice along the Y-axis at the given position
                    sdf_slice = sdf_pyvista_grid.slice(normal='y', origin=(sdf_pyvista_grid.center[0], y_pos, sdf_pyvista_grid.center[2]))

                    # If the slice is outside the object, it might be empty.
                    if sdf_slice.n_points == 0:
                        return # Do nothing, leaving the dark grey background

                    # Add the solid-filled regions using n_colors for a binned effect
                    p3.add_mesh(
                        sdf_slice,
                        name="slice_mesh",
                        scalars='sdf',
                        cmap='viridis',
                        n_colors=num_bins, # Use number of bins for discrete colors
                        clim=[sdf_min, sdf_max], # Enforce the color limits
                        show_scalar_bar=False,
                    )

                    # Generate and add subtle contour lines that match the bins
                    edges = sdf_slice.contour(contour_levels, scalars='sdf')
                    p3.add_mesh(edges, name="slice_edges", color='white', line_width=1, opacity=0.4)

                # Manually call the update function once to populate the plotter before adding the scalar bar
                initial_y_pos = sdf_pyvista_grid.center[1]
                update_slice_y(initial_y_pos)

                # Now that a mesh exists, add a scalar bar that respects the binned colors
                p3.add_scalar_bar(
                    title='SDF Value',
                    n_labels=5, # Approximate number of labels
                    color='white'
                )

                # Add a slider to control the slice position along the Y-axis
                y_bounds = sdf_pyvista_grid.bounds[2:4]
                p3.add_slider_widget(
                    callback=update_slice_y,
                    rng=y_bounds,
                    value=initial_y_pos,
                    title="Y Slice Position",
                    style='modern'
                )

                p3.add_text("Interactive SDF Slice along Y-axis", font_size=12)
                p3.view_xz() # Set camera to view the XZ plane (correct for a Y-slice)
                p3.show(title=f"Interactive SDF Slice for {instance_name}")

        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]  # last entry
            print(f"  [ERROR] Failed to process {instance_name}: {e} (line {tb.lineno})")



# def _worker(instance_path, root):
#     instance_name = os.path.basename(instance_path)
#     if instance_name in SKIP:
#         return instance_name, "skipped"

#     try:
#         post_adjustment_sdf_mod(
#             instance_name,
#             root,
#             save_bool=True,
#             save_folder=os.path.join(root, "data_pose"),
#             visualize=False,
#         )
#         return instance_name, "ok"
#     except Exception as e:
#         # Return error rather than crashing the whole pool
#         return instance_name, f"error: {e}"

SKIP = {
    "2e35f84025d83fcd1d1eb082bbbd4dc8b991ab501d81eae7a8a710012a58e38a",
    "118519d1320c7f18f2fad814d8d05337ac12f6d10e9e40946c40b60c94d36b25",
    "6a9fc414a95022786e905694939ddee30f10bbc438153a10a7dc6dda11993a6f",
    "56a370e60eebd93a0aeb9426c2636ce54172d1b212f979c23006c6ed902c58d2",
    "27623ddea9ef189b888b60917c113724e5169bc690acc56b8293332b3729381f",
    "348af3ce6e523d27e557f9b5b53ba5f6953af1595b67995a933fdea19e2cf523",
    "2f54b1754daf4de74e791c0ca67220d84f7cde4ef3dfaadaab00a37d9dd1989f",
    "40324e56315d7af575eec7b92bc581a85e72a4caeb03ce9f22ae25a3c340ae21",
    "0c22903d7ee27d5b3ca11481d7ea68647e27cdd267eee621b9b4ee5a5bbd630c"
    #"67f94729ae5b4a211b1a7db0444e29d2d098bc1e6fab356bb25afce98b702a16"
}
# def _init_worker():
#     # Let the main process handle Ctrl+C; workers ignore SIGINT
#     signal.signal(signal.SIGINT, signal.SIG_IGN)

# def _worker(instance_path, root):
#     instance_name = os.path.basename(instance_path)
#     if instance_name in SKIP:
#         return instance_name, "skipped"

#     post_adjustment_sdf_mod(
#         instance_name,
#         root,
#         save_bool=True,
#         save_folder=os.path.join(root, "data_pose"),
#         visualize=False,
#     )
#     return instance_name, "ok"

# def run_all(instances, root, max_workers=None):
#     # On some platforms this helps avoid weird signal behavior
#     try:
#         mp.set_start_method("spawn")
#     except RuntimeError:
#         pass  # already set

#     todo = [p for p in instances if os.path.basename(p) not in SKIP]

#     ex = ProcessPoolExecutor(
#         max_workers=max_workers,
#         initializer=_init_worker,
#     )

#     futures = [ex.submit(_worker, p, root) for p in todo]

#     try:
#         for fut in as_completed(futures):
#             name, status = fut.result()
#             print(f"{name}: {status}", flush=True)

#     except KeyboardInterrupt:
#         print("\n^C received — cancelling pending tasks and terminating workers...", flush=True)
#         # Cancel anything not started yet
#         for f in futures:
#             f.cancel()
#         # Stop the pool immediately; don't wait
#         ex.shutdown(wait=False, cancel_futures=True)
#         # Belt-and-suspenders: kill any leftover children
#         for child in mp.active_children():
#             try:
#                 child.kill()  # send SIGKILL
#             except Exception:
#                 pass
#         raise SystemExit(130)

#     except Exception as e:
#         # Make sure we don't hang on other errors either
#         ex.shutdown(wait=False, cancel_futures=True)
#         raise

#     else:
#         ex.shutdown(wait=True)  # clean shutdown when all done


# TIMEOUT_S = 900  # per-task wall time (optional)

# def _init_worker():
#     # keep Ctrl+C in main only and tame BLAS threads
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("MKL_NUM_THREADS", "1")
#     os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#     os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# def _worker(instance_path, root):
#     name = os.path.basename(instance_path)
#     if name in SKIP:
#         return name, "skipped"

#     # Do the work; let exceptions propagate (we'll catch them in main)
#     post_adjustment_sdf_mod(
#         name,
#         root,
#         save_bool=True,
#         save_folder=os.path.join(root, "data_pose"),
#         visualize=False,
#     )
#     return name, "ok"


# def process_all(instances, root, failures_path="failures.jsonl", successes_path="successes.txt", max_workers=None):
#     try:
#         mp.set_start_method("spawn")
#     except RuntimeError:
#         pass

#     todo = [p for p in instances if os.path.basename(p) not in SKIP]

#     # Ensure parent dirs exist if you pass something like logs/failures.jsonl
#     os.makedirs(os.path.dirname(failures_path) or ".", exist_ok=True)
#     os.makedirs(os.path.dirname(successes_path) or ".", exist_ok=True)

#     f_fail = open(failures_path or "failures.jsonl", "a", buffering=1)
#     f_ok   = open(successes_path or "successes.txt",  "a", buffering=1)

#     def log_failure(name, err, tb):
#         rec = {"instance": name, "error": repr(err), "traceback": tb}
#         f_fail.write(json.dumps(rec) + "\n"); f_fail.flush()

#     def log_success(name, status):
#         f_ok.write(f"{name}\t{status}\n"); f_ok.flush()

#     with ProcessPoolExecutor(max_workers=max_workers or (os.cpu_count() or 4),
#                              initializer=_init_worker) as ex:
#         futures = {ex.submit(_worker, p, root): os.path.basename(p) for p in todo}
#         try:
#             for fut in as_completed(futures):
#                 name = futures[fut]
#                 try:
#                     res_name, status = fut.result(timeout=TIMEOUT_S)
#                     log_success(res_name, status)
#                     print(f"{res_name}: {status}", flush=True)
#                 except Exception as e:
#                     tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
#                     log_failure(name, e, tb)
#                     print(f"{name}: ERROR -> {e}", flush=True)
#         except KeyboardInterrupt:
#             print("\n^C — cancelling pending tasks...", flush=True)
#             for f in futures: f.cancel()
#             ex.shutdown(wait=False, cancel_futures=True)
#             for child in mp.active_children():
#                 try: child.kill()
#                 except Exception: pass
#             raise SystemExit(130)
#         finally:
#             f_fail.close(); f_ok.close()

# Per-task wall clock timeout (seconds). Tune to your workload.

NPROC         = 32
TIMEOUT_S     = 900          # per-item wall time; set 0/None to disable
MAKTASKS      = 50           # tasks per worker before recycle (speed vs isolation)
CHUNKSIZE     = 8            # batch items per IPC round
MAX_RETRIES   = 1
class _Timeout(Exception): pass

def _init_pool():
    # keep SIGINT in main; clamp BLAS threads
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _alarm_handler(signum, frame):
    raise _Timeout()

def _run_one(args):
    """Worker: returns a structured dict for success or failure."""
    instance_path, root = args
    name = os.path.basename(instance_path)

    # Optional quick prechecks
    if name in SKIP:
        return {"name": name, "status": "skipped"}
    if not os.path.exists(instance_path):
        return {"name": name, "status": "error", "exc_type": "FileNotFoundError",
                "exc_msg": f"path not found: {instance_path}", "traceback": None}

    # Set per-task timeout (Unix/macOS). On Windows, SIGALRM isn't available.
    if TIMEOUT_S:
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(TIMEOUT_S)
        except Exception:
            pass  # platform without SIGALRM

    try:
        post_adjustment_sdf_mod(
            name, root,
            save_bool=True,
            save_folder=os.path.join(root, "data_pose_norm"),
            visualize=False,
        )
        return {"name": name, "status": "ok"}

    except _Timeout:
        return {"name": name, "status": "timeout", "exc_type": "Timeout",
                "exc_msg": f">{TIMEOUT_S}s", "traceback": None}

    except Exception as e:
        # Capture full details
        return {"name": name, "status": "error",
                "exc_type": type(e).__name__,
                "exc_msg": str(e),
                "traceback": traceback.format_exc()}

    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass

def process_all_fast(instances, root,
                     failures_path="failures.jsonl",
                     successes_path="successes.txt"):
    ctx = mp.get_context("spawn")
    todo = [(p, 0) for p in instances]  # (path, retries)

    os.makedirs(os.path.dirname(failures_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(successes_path) or ".", exist_ok=True)

    with open(failures_path, "a", buffering=1) as f_fail, \
         open(successes_path, "a", buffering=1) as f_ok, \
         ctx.Pool(processes=NPROC, initializer=_init_pool, maxtasksperchild=MAKTASKS) as pool:

        def log_failure(rec):
            f_fail.write(json.dumps(rec) + "\n"); f_fail.flush()

        def log_success(name, status):
            f_ok.write(f"{name}\t{status}\n"); f_ok.flush()

        # submit all at once; chunking reduces IPC overhead
        args = [(p, root) for p, _ in todo]
        pending = pool.imap_unordered(_run_one, args, chunksize=CHUNKSIZE)

        # map for retries
        path_by_name = {os.path.basename(p): p for p, _ in todo}
        retry_counts = {os.path.basename(p): r for p, r in todo}

        for rec in pending:
            name = rec["name"]
            status = rec["status"]

            if status in ("ok", "skipped"):
                print(f"{name}: {status}", flush=True)
                log_success(name, status)
                continue

            # Failure path: print meaningful info and log JSONL
            exc_type = rec.get("exc_type")
            exc_msg = rec.get("exc_msg")
            print(f"{name}: {status} [{exc_type}] {exc_msg}", flush=True)
            log_failure(rec)

            # Retry logic (for transient errors)
            tries = retry_counts.get(name, 0)
            if tries < MAX_RETRIES:
                retry_counts[name] = tries + 1
                pool.apply_async(_run_one, ((path_by_name[name], root),),
                                 callback=lambda r: pending._items.append(r))  # lightweight requeue




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superimpose SDF and voxel grid.")
    parser.add_argument("--resolution", type=int, default=64, help="Resolution of the SDF grid.")
    parser.add_argument("--min_efficiency", type=float, default=0.15, help="Minimum bounding box efficiency for mesh density.")
    parser.add_argument("--pitch", type=float, default=0.01, help="Pitch for bounding box efficiency calculation.")
    parser.add_argument("--root", type=str, default='/home/user/TRELLIS/datasets/ObjaverseXL_sketchfab', help="Root directory for instances.")

    # SKIP = {
    # "2e35f84025d83fcd1d1eb082bbbd4dc8b991ab501d81eae7a8a710012a58e38a",
    # "118519d1320c7f18f2fad814d8d05337ac12f6d10e9e40946c40b60c94d36b25",
    # }

    args = parser.parse_args()

    #resolution = args.resolution
    #print(f"Using resolution: {resolution}")
    instances = glob.glob(os.path.join(args.root, "renders_cond") + "/*")
    todo = [p for p in instances if os.path.basename(p) not in SKIP]
    max_workers = 16
    #run_all(instances, args.root, max_workers)
    process_all_fast(instances, args.root, failures_path="bad_instances.jsonl", successes_path="done.txt")


    # with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #     futures = [ex.submit(_worker, p, args.root) for p in todo]
    #     for fut in as_completed(futures):
    #         name, status = fut.result()
    #         print(f"{name}: {status}")
    # for instance in todo:
    #     instance_name = os.path.basename(instance)
    # #     if instance_name == '38dae551176a8f8f2be3332fc249348684659e0ba87f88082d2b6dbd5770432c':
    # #         post_adjustment_sdf_mod(instance_name, args.root, save_bool=True, save_folder=os.path.join(args.root,'data_pose_norm'), visualize=False)

    #     # if instance_name =='2e35f84025d83fcd1d1eb082bbbd4dc8b991ab501d81eae7a8a710012a58e38a' or instance_name== '118519d1320c7f18f2fad814d8d05337ac12f6d10e9e40946c40b60c94d36b25':
    #     #    continue
    #     print(instance_name)
    #     post_adjustment_sdf_mod(instance_name, args.root, save_bool=True, save_folder=os.path.join(args.root,'data_pose_norm'), visualize=False)
