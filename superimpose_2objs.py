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
from PIL import Image, ImageOps
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

# ---------- step 1: canonical high-res SDF ----------
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

def canonical_sdf_u2(mesh_path, res0=128):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    V = m.vertices.view(np.ndarray).astype(np.float64)
    # print("Pre-normalization AABB:", V.min(0), V.max(0))
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
    frame_id=None, margin_vox=1,
    space="unitcube"  # "unitcube" -> [-0.5,0.5]^3 ; "unit2" -> [-1,1]^3
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
    os.makedirs(os.path.join(out_dir, "voxels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "occs"),   exist_ok=True)
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
    utils3d.io.write_ply(os.path.join(out_dir, "voxels", f"{name}.ply"), centers.astype(np.float32))

    np.save(os.path.join(out_dir, "occs", f"{name}.npy"), occ)
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
def save_scene_sdfs_only(
    out_dir, scene_name,
    items,                   # dict: { mesh_key: {"sdf": ndarray, "occ": ndarray or None} }
    *,                       # mesh_key e.g. "hand", "object"
    res=64,
    space="unit2",           # "unit2" → [-1,1]^3 grid
    R_fixed=None, s_aug=None, t_aug=None, c0=None,
    frame_id=None, margin_vox=1,
    extra_meta: dict | None = None,
    contact_mask: np.ndarray | None = None,
    dist_to_contact: np.ndarray | None = None,
    dx: float | None = None,     # spacing used for dist_to_contact (e.g. 2.0/64)
):
    """
    Save a SINGLE meta JSON for the scene + per-mesh SDF and indices,
    optionally also saving:
      - contact_mask (bool [R,R,R])
      - contact_coords ([N,3] int indices)
      - dist_to_contact (float [R,R,R]) in same metric as SDF.

    Writes:
      sdfs/{scene}__{mesh}.npy
      idxs/{scene}__{mesh}.npy
      contacts/{scene}_contact_mask.npy          (if contact_mask given)
      contacts/{scene}_contact_coords.npy        (if contact_mask given)
      contacts/{scene}_dist_to_contact.npy       (if dist_to_contact given)
      {scene}_meta.json
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sdfs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "idxs"), exist_ok=True)

    # --- optional contacts dir ---
    has_contacts = (contact_mask is not None) or (dist_to_contact is not None)
    if has_contacts:
        os.makedirs(os.path.join(out_dir, "contacts"), exist_ok=True)

    # --- grid convention (shared) ---
    if space == "unitcube":
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        voxel_size = 1.0 / res
    elif space == "unit2":
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        voxel_size = 2.0 / res
    else:
        raise ValueError(f"Unknown space '{space}'")

    # --- write per-mesh SDF + idx (no occs, no voxels) ---
    stats = {}
    for key, payload in items.items():
        sdf = np.asarray(payload["sdf"], dtype=np.float32)
        assert sdf.shape == (res, res, res), f"{key}: bad sdf shape {sdf.shape}"

        # choose indices: prefer provided occ, else sdf<=0
        occ = payload.get("occ", None)
        if occ is not None:
            occ = np.asarray(occ).astype(bool)
            assert occ.shape == (res, res, res), f"{key}: bad occ shape {occ.shape}"
            idx = np.argwhere(occ)
        else:
            idx = np.argwhere(sdf <= 0.0)

        np.save(os.path.join(out_dir, "sdfs", f"{scene_name}__{key}.npy"), sdf)
        np.save(os.path.join(out_dir, "idxs", f"{scene_name}__{key}.npy"), idx.astype(np.int32))

        stats[key] = {
            "n_idx": int(idx.shape[0]),
            "sdf_min": float(sdf.min()),
            "sdf_max": float(sdf.max()),
            "sdf_mean": float(sdf.mean()),
        }

    # --- optional: contacts ---
    contacts_info = None
    if has_contacts:
        contacts_info = {}

        if contact_mask is not None:
            # contact_mask = np.asarray(contact_mask, dtype=bool)
            assert contact_mask.shape == (res, res, res), "contact_mask has wrong shape"

            # contact_mask_path = os.path.join(out_dir, "contacts", f"{scene_name}_contact_mask.npy")
            # np.save(contact_mask_path, contact_mask)

            contact_coords = np.argwhere(contact_mask)
            contact_coords_path = os.path.join(out_dir, "contacts", f"{scene_name}_contact_coords.npy")
            np.save(contact_coords_path, contact_coords.astype(np.int32))

            contacts_info["n_contacts"] = int(contact_coords.shape[0])
            # contacts_info["contact_mask"] = contact_mask_path
            contacts_info["contact_coords"] = contact_coords_path

        if dist_to_contact is not None:
            dist_to_contact = np.asarray(dist_to_contact, dtype=np.float32)
            assert dist_to_contact.shape == (res, res, res), "dist_to_contact has wrong shape"

            dist_path = os.path.join(out_dir, "contacts", f"{scene_name}_dist_to_contact.npy")
            np.save(dist_path, dist_to_contact)

            contacts_info["dist_to_contact"] = dist_path
            contacts_info["dist_min"] = float(dist_to_contact.min())
            contacts_info["dist_max"] = float(dist_to_contact.max())
            contacts_info["dist_mean"] = float(dist_to_contact.mean())
            if dx is not None:
                contacts_info["dist_sampling_dx"] = float(dx)

    # --- shared meta (only once) ---
    meta = {
        "grid": {
            "res": int(res),
            "space": space,
            "origin": origin.tolist(),
            "voxel_size": float(voxel_size),
            "bbox_min": origin.tolist(),
            "bbox_max": (origin + voxel_size * res).tolist(),
            "index_axes": "i->x, j->y, k->z",
            "indexing": "np.meshgrid(..., indexing='ij')",
            "sdf_negative_inside": True,
            "iso_value": 0.0
        },
        "pose": {
            "R_fixed": None if R_fixed is None else np.asarray(R_fixed, dtype=float).tolist(),
            "s_aug": None if s_aug is None else float(s_aug),
            "t_aug": None if t_aug is None else np.asarray(t_aug, dtype=float).tolist(),
            "c0":    None if c0    is None else np.asarray(c0,    dtype=float).tolist(),
            "about": "similarity about shared pivot c0"
        },
        "provenance": {
            "frame_id": None if frame_id is None else int(frame_id),
            "margin_vox": int(margin_vox),
            "pipeline": "mesh2sdf->resample->(optional)skfmm(+contacts)",
        },
        "stats_per_mesh": stats,
    }

    if contacts_info is not None:
        meta["contacts"] = contacts_info

    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(out_dir, f"{scene_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # return paths for convenience
    paths = {
        key: {
            "sdf": os.path.join(out_dir, "sdfs", f"{scene_name}__{key}.npy"),
            "idx": os.path.join(out_dir, "idxs", f"{scene_name}__{key}.npy"),
        }
        for key in items.keys()
    }
    if contacts_info is not None:
        paths["contacts"] = {}
        if "contact_mask" in contacts_info:
            paths["contacts"]["mask"]   = contacts_info["contact_mask"]
            paths["contacts"]["coords"] = contacts_info["contact_coords"]
        if "dist_to_contact" in contacts_info:
            paths["contacts"]["dist"] = contacts_info["dist_to_contact"]

    paths["meta"] = os.path.join(out_dir, f"{scene_name}_meta.json")
    return paths


def save_scene_sdfs_only_correct(
    out_dir, scene_name,
    items,                   # dict: { mesh_key: {"sdf": ndarray, "occ": ndarray or None} }
    *,                       # mesh_key e.g. "hand", "object"
    res=64,
    space="unit2",           # "unit2" → [-1,1]^3 grid
    R_fixed=None, s_aug=None, t_aug=None, c0=None,
    frame_id=None, margin_vox=1,
    extra_meta: dict | None = None,
    contact_mask: np.ndarray | None = None,
    dist_to_contact: np.ndarray | None = None,
    dx: float | None = None,     # spacing used for dist_to_contact (e.g. 2.0/64)
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sdfs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "idxs"), exist_ok=True)

    has_contacts = (contact_mask is not None) or (dist_to_contact is not None)
    if has_contacts:
        os.makedirs(os.path.join(out_dir, "contacts"), exist_ok=True)

    # --- grid convention (shared) ---
    if space == "unitcube":
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        voxel_size = 1.0 / res
    elif space == "unit2":
        origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        voxel_size = 2.0 / res
    else:
        raise ValueError(f"Unknown space '{space}'")

    stats = {}
    union_occ = None   # <--- NEW: will store hand ∪ object occupancy

    # --- write per-mesh SDF + idx (no occs, no voxels) ---
    for key, payload in items.items():
        sdf = np.asarray(payload["sdf"], dtype=np.float32)
        assert sdf.shape == (res, res, res), f"{key}: bad sdf shape {sdf.shape}"

        occ = payload.get("occ", None)
        if occ is not None:
            occ = np.asarray(occ).astype(bool)
            assert occ.shape == (res, res, res), f"{key}: bad occ shape {occ.shape}"
            idx = np.argwhere(occ)

            # --- NEW: build union occupancy over all provided occs ---
            if union_occ is None:
                union_occ = occ.copy()
            else:
                union_occ |= occ
        else:
            # fallback: indices from sdf<=0 if no occ given
            idx = np.argwhere(sdf <= 0.0)

        np.save(os.path.join(out_dir, "sdfs", f"{scene_name}__{key}.npy"), sdf)
        np.save(os.path.join(out_dir, "idxs", f"{scene_name}__{key}.npy"), idx.astype(np.int32))

        stats[key] = {
            "n_idx": int(idx.shape[0]),
            "sdf_min": float(sdf.min()),
            "sdf_max": float(sdf.max()),
            "sdf_mean": float(sdf.mean()),
        }

    # --- optional: contacts ---
    contacts_info = None
    if has_contacts:
        contacts_info = {}
        if contact_mask is not None:
            assert contact_mask.shape == (res, res, res), "contact_mask has wrong shape"
            contact_coords = np.argwhere(contact_mask)
            contact_coords_path = os.path.join(out_dir, "contacts", f"{scene_name}_contact_coords.npy")
            np.save(contact_coords_path, contact_coords.astype(np.int32))
            contacts_info["n_contacts"] = int(contact_coords.shape[0])
            contacts_info["contact_coords"] = contact_coords_path

        if dist_to_contact is not None:
            dist_to_contact = np.asarray(dist_to_contact, dtype=np.float32)
            assert dist_to_contact.shape == (res, res, res), "dist_to_contact has wrong shape"
            dist_path = os.path.join(out_dir, "contacts", f"{scene_name}_dist_to_contact.npy")
            np.save(dist_path, dist_to_contact)
            contacts_info["dist_to_contact"] = dist_path
            contacts_info["dist_min"] = float(dist_to_contact.min())
            contacts_info["dist_max"] = float(dist_to_contact.max())
            contacts_info["dist_mean"] = float(dist_to_contact.mean())
            if dx is not None:
                contacts_info["dist_sampling_dx"] = float(dx)

    # --- NEW: pose2d_meta from union occ (hand ∪ object) ---
    pose2d_meta = None
    if union_occ is not None:
        pose2d_meta = pose2d_meta_from_occ(union_occ, res=res)
    # if union_occ is None, pose2d_meta stays None

    # --- shared meta (only once) ---
    meta = {
        "grid": {
            "res": int(res),
            "space": space,
            "origin": origin.tolist(),
            "voxel_size": float(voxel_size),
            "bbox_min": origin.tolist(),
            "bbox_max": (origin + voxel_size * res).tolist(),
            "index_axes": "i->x, j->y, k->z",
            "indexing": "np.meshgrid(..., indexing='ij')",
            "sdf_negative_inside": True,
            "iso_value": 0.0
        },
        "pose": {
            "R_fixed": None if R_fixed is None else np.asarray(R_fixed, dtype=float).tolist(),
            "s_aug": None if s_aug is None else float(s_aug),
            "t_aug": None if t_aug is None else np.asarray(t_aug, dtype=float).tolist(),
            "c0":    None if c0    is None else np.asarray(c0,    dtype=float).tolist(),
            "about": "similarity about shared pivot c0"
        },
        "provenance": {
            "frame_id": None if frame_id is None else int(frame_id),
            "margin_vox": int(margin_vox),
            "pipeline": "mesh2sdf->resample->(optional)skfmm(+contacts)",
        },
        "stats_per_mesh": stats,
    }

    if pose2d_meta is not None:
        meta["pose2d_meta"] = pose2d_meta

    if contacts_info is not None:
        meta["contacts"] = contacts_info

    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(out_dir, f"{scene_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- return paths ---
    paths = {
        key: {
            "sdf": os.path.join(out_dir, "sdfs", f"{scene_name}__{key}.npy"),
            "idx": os.path.join(out_dir, "idxs", f"{scene_name}__{key}.npy"),
        }
        for key in items.keys()
    }

    if contacts_info is not None:
        paths["contacts"] = {}
        # NOTE: we only write coords + dist; no mask file
        if "contact_coords" in contacts_info:
            paths["contacts"]["coords"] = contacts_info["contact_coords"]
        if "dist_to_contact" in contacts_info:
            paths["contacts"]["dist"] = contacts_info["dist_to_contact"]

    paths["meta"] = os.path.join(out_dir, f"{scene_name}_meta.json")
    return paths




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

def xfov_to_intrinsics(cam_angle_x, W, H):
    fx = 0.5*W / np.tan(0.5*float(cam_angle_x))
    fy = fx * (W/H)
    cx, cy = 0.5*W, 0.5*H
    return fx, fy, cx, cy

def project_points_u2(Pw_u2, c2w, fx, fy, cx, cy):
    w2c = np.linalg.inv(c2w)
    Pw_h = np.concatenate([Pw_u2, np.ones((Pw_u2.shape[0],1))], axis=1)
    Pc   = (w2c @ Pw_h.T).T[:, :3]
    x, y, z = Pc[:,0], Pc[:,1], Pc[:,2]
    z = np.clip(z, 1e-8, None)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=1)

def bbox2d(uv):
    u, v = uv[:,0], uv[:,1]
    umin, umax = np.min(u), np.max(u)
    vmin, vmax = np.min(v), np.max(v)
    cx, cy = 0.5*(umin+umax), 0.5*(vmin+vmax)
    side   = max(umax-umin, vmax-vmin)
    return cx, cy, side

def crop_image_pose_faithful(
    image_rgba: Image.Image,
    V_u2_can: np.ndarray,        # 2*s_blend*(V_raw - c_blend)
    V_u2_pose: np.ndarray,       # ((V_u2_can - c0) @ R) * s + t
    c2w: np.ndarray,             # frame’s 4x4
    cam_angle_x: float,          # radians
    out_size: int = 512,
    pad_ratio: float = 1.15,
    min_crop_px: int = 8         # small so scale difference survives
):
    assert image_rgba.mode == 'RGBA'
    W, H = image_rgba.size
    fx, fy, cx, cy = xfov_to_intrinsics(cam_angle_x, W, H)

    uv_can  = project_points_u2(V_u2_can,  c2w, fx, fy, cx, cy)
    uv_pose = project_points_u2(V_u2_pose, c2w, fx, fy, cx, cy)

    c_can_x, c_can_y, s_can = bbox2d(uv_can)
    c_pos_x, c_pos_y, s_pos = bbox2d(uv_pose)

    # We WANT the crop to reflect translation & scale:
    cx_crop = c_pos_x
    cy_crop = c_pos_y
    crop_side = pad_ratio * s_pos

    # Bounds (no “normalize away” heuristics)
    crop_side = np.clip(crop_side, min_crop_px, max(W, H))
    half = 0.5 * crop_side
    L, R = cx_crop - half, cx_crop + half
    T, B = cy_crop - half, cy_crop + half

    # clamp to image, but do not change crop_side unless forced
    if (L < 0) or (T < 0) or (R > W) or (B > H):
        L = max(L, 0); T = max(T, 0); R = min(R, W); B = min(B, H)

    # If degenerate (fully out of bounds), fall back to full image
    if (R <= L) or (B <= T):
        L, T, R, B = 0, 0, W, H

    L, T, R, B = int(np.floor(L)), int(np.floor(T)), int(np.ceil(R)), int(np.ceil(B))
    crop = image_rgba.crop((L, T, R, B)).resize((out_size, out_size), Image.Resampling.LANCZOS)

    # premultiply alpha
    a = np.array(crop.getchannel('A'), dtype=np.float32)/255.0
    rgb_np = np.array(crop.convert('RGB'), dtype=np.float32)/255.0
    rgb_np = (rgb_np * a[...,None])
    rgb = Image.fromarray((rgb_np*255.0).astype(np.uint8))

    debug = {
        "W": W, "H": H,
        "s_can_px": float(s_can),
        "s_pos_px": float(s_pos),
        "scale_ratio_px": float(s_pos / max(s_can, 1e-6)),
        "can_center_px": [float(c_can_x), float(c_can_y)],
        "pos_center_px": [float(c_pos_x), float(c_pos_y)],
        "crop_box": [L, T, R, B],
        "crop_side_px": float(crop_side),
    }
    return rgb, debug

def make_grid(res=64, unit="unit2"):
    """
    Build a full-domain grid with POINT dimensions = (res+1)^3.
    unit='unit2' -> domain [-1,1]^3, spacing = 2/res
    unit='unit1' -> domain [-0.5,0.5]^3, spacing = 1/res
    Returns pv.ImageData (works on old PyVista too).
    """
    g = pv.ImageData()
    if unit == "unit2":
        origin  = (-1.0, -1.0, -1.0)
        spacing = (2.0/res, 2.0/res, 2.0/res)
    else:  # unit1
        origin  = (-0.5, -0.5, -0.5)
        spacing = (1.0/res, 1.0/res, 1.0/res)

    # IMPORTANT: point dimensions are cells+1
    g.dimensions = (res+1, res+1, res+1)
    g.origin     = origin
    g.spacing    = spacing
    return g

def field_to_point_grid(grid, name, arr_3d):
    """
    Attach arr_3d (shape [R,R,R], per-voxel) as CELL data,
    then convert to POINT data for contouring.
    """
    g = grid.copy()
    g.cell_data[name] = np.asarray(arr_3d).ravel(order='F')  # one value per voxel
    return g.cell_data_to_point_data()  # now 'name' is point data


def image_example(image, image_size=518):

    alpha = np.array(image.getchannel(3))
    bbox = np.array(alpha).nonzero()
    bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_size_ratio = 1.2
    aug_hsize = hsize * aug_size_ratio
    aug_center_offset = [0, 0]
    aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
    aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
    image = image.crop(aug_bbox)

    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = image.getchannel(3)
    image = image.convert('RGB')
    return image

# def collage_from_meta_bbox_preserve_aspect(
#     image_rgba: Image.Image,
#     pose2d_meta: dict,
#     out_size: int = 518,
# ):
#     """
#     Place the RGBA sprite into a square canvas so its size+translation in the
#     image plane matches the saved grid bbox (no stretching; uniform scale).
#     """
#     assert image_rgba.mode == "RGBA"
#     W = H = int(out_size)
#     res      = int(pose2d_meta["res"])
#     bbox_xy  = pose2d_meta["bbox_xy"]
#     x_right  = bool(pose2d_meta.get("x_right", True))
#     y_up     = bool(pose2d_meta.get("y_up", True))

#     canvas = Image.new("RGBA", (W, H), (0,0,0,0))
#     if bbox_xy is None:
#         return canvas  # empty mask case

#     xmin, ymin, xmax, ymax = bbox_xy
#     # convert voxel bbox → pixel bbox
#     px_per_vox = float(W) / float(res)
#     xmin_px = int(round(xmin * px_per_vox))
#     ymin_px = int(round(ymin * px_per_vox))
#     w_px_box = max(1, int(round((xmax - xmin + 1) * px_per_vox)))
#     h_px_box = max(1, int(round((ymax - ymin + 1) * px_per_vox)))

#     # crop sprite from Blender alpha (original render content)
#     A = np.asarray(image_rgba.getchannel("A"))
#     ys, xs = np.nonzero(A > 0)
#     if len(xs) == 0:
#         return canvas
#     x0, x1 = xs.min(), xs.max()
#     y0, y1 = ys.min(), ys.max()
#     sprite = image_rgba.crop((x0, y0, x1+1, y1+1))
#     sw, sh = sprite.size

#     # uniform scale to fit inside target box (preserve aspect)
#     scale = max(1e-6, min(w_px_box / sw, h_px_box / sh))
#     new_w = max(1, int(round(sw * scale)))
#     new_h = max(1, int(round(sh * scale)))
#     sprite_resized = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

#     # center within the target box
#     dx_px = (w_px_box - new_w) // 2
#     dy_px = (h_px_box - new_h) // 2
#     paste_x = xmin_px + dx_px
#     paste_y = ymin_px + dy_px

#     # If you ever need to invert translation conventions post-hoc,
#     # you can modify (paste_x, paste_y) here using x_right/y_up flags.
#     # (We already encoded those when producing bbox_xy, so no flips now.)

#     paste_x = max(-new_w, min(W, paste_x))
#     paste_y = max(-new_h, min(H, paste_y))
#     canvas.alpha_composite(sprite_resized, dest=(paste_x, paste_y))
#     return canvas

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
    canvas = Image.new("RGBA", (W, H), (0,0,0,255))
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



def collage_from_union_indices(
    image_rgba: Image.Image,
    idx_list: list,          # [idx_hand, idx_object, ...], each shape [N,3] int (i,j,k)
    res: int,                # grid resolution from shared meta["grid"]["res"]
    out_size: int = 1024,
    pad_ratio: float = 0.10, # extra margin around union bbox (fraction of max(w,h) in voxels)
):
    """
    Build a square collage where the RGBA sprite is uniformly scaled and
    positioned so that its image-plane footprint matches the union of the
    provided voxel indices (hand ∪ object).
    """
    assert image_rgba.mode == "RGBA"
    W = H = int(out_size)
    canvas = Image.new("RGBA", (W, H), (0,0,0,0))

    # --- union bbox in voxel coords (x=i, y=j) ---
    valid_idxs = [np.asarray(idx, dtype=int) for idx in idx_list if idx is not None and len(idx)]
    if not valid_idxs:
        return canvas  # nothing to show

    all_idx = np.vstack(valid_idxs)
    xmin = int(all_idx[:,0].min()); xmax = int(all_idx[:,0].max())
    ymin = int(all_idx[:,1].min()); ymax = int(all_idx[:,1].max())

    # pad in voxels (proportional to the larger side)
    w_vox = xmax - xmin + 1
    h_vox = ymax - ymin + 1
    pad_vox = int(round(max(w_vox, h_vox) * pad_ratio))
    xmin = max(0, xmin - pad_vox)
    ymin = max(0, ymin - pad_vox)
    xmax = min(res - 1, xmax + pad_vox)
    ymax = min(res - 1, ymax + pad_vox)

    # --- voxel bbox -> pixel bbox in output square ---
    px_per_vox = float(W) / float(res)
    xmin_px = int(round(xmin * px_per_vox))
    ymin_px = int(round(ymin * px_per_vox))
    w_px_box = max(1, int(round((xmax - xmin + 1) * px_per_vox)))
    h_px_box = max(1, int(round((ymax - ymin + 1) * px_per_vox)))

    # --- crop the sprite (alpha mask) from the input RGBA ---
    A = np.asarray(image_rgba.getchannel("A"))
    ys, xs = np.nonzero(A > 0)
    if len(xs) == 0:
        return canvas
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    sprite = image_rgba.crop((x0, y0, x1+1, y1+1))
    sw, sh = sprite.size

    # --- uniform scale to fit inside target union box ---
    scale = max(1e-6, min(w_px_box / sw, h_px_box / sh))
    new_w = max(1, int(round(sw * scale)))
    new_h = max(1, int(round(sh * scale)))
    sprite_resized = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # --- center within the union box ---
    paste_x = xmin_px + (w_px_box - new_w)//2
    paste_y = ymin_px + (h_px_box - new_h)//2
    paste_x = max(-new_w, min(W, paste_x))
    paste_y = max(-new_h, min(H, paste_y))

    canvas.alpha_composite(sprite_resized, dest=(paste_x, paste_y))
    return canvas

import open3d as o3d

def load_vertices_trimesh_with_fix(mesh_path, R_x90):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    V_raw = m.vertices.view(np.ndarray).astype(np.float64)
    F     = m.faces.view(np.ndarray).astype(np.int32)
    V_fix = V_raw @ R_x90.T  # row-vector convention → column v' = R_x90 v
    return V_fix, F


def compute_shared_normalizer_unit2_trimesh(
    mesh_paths,
    R_x90,
    pad_frac=0.01,
    sphere_index=None,
    force_sphere_z0=False   # <- default False
):
    Vs = []
    for p in mesh_paths:
        m = trimesh.load(p, force='mesh', process=False)
        V_raw = m.vertices.view(np.ndarray).astype(np.float64)
        V_fix = V_raw @ R_x90.T
        Vs.append(V_fix)

    V_all = np.vstack(Vs)
    vmin = V_all.min(axis=0)
    vmax = V_all.max(axis=0)

    center = 0.5 * (vmin + vmax)
    extent_long = (vmax - vmin).max()
    extent_long = max(extent_long * (1.0 + pad_frac), 1e-9)

    s = 2.0 / extent_long
    z_shift = 0.0          # <- no z-shift in canonical

    return s, center, z_shift


def canonical_sdf_u2_with_shared_norm_trimesh(
    mesh_path,
    s,
    center,
    R_x90,
    z_shift=0.0,
    res0=128
):
    V_fix, F = load_vertices_trimesh_with_fix(mesh_path, R_x90)
    V_u2 = apply_normalizer_unit2(V_fix, s, center, z_shift)

    # debugging: check the box
    # print("AABB in unit2:", V_u2.min(0), V_u2.max(0))

    V_u2 = np.clip(V_u2, -1 + 1e-6, 1 - 1e-6)

    level = 2.0 / res0
    sdf0_raw, _ = mesh2sdf.compute(V_u2, F, res0, fix=False, level=level, return_mesh=True)
    sdf0 = np.asarray(sdf0_raw).reshape(res0, res0, res0)
    return V_u2, F, sdf0



def compute_shared_normalizer_unit2(
    mesh_paths,
    transforms=None,
    pad_frac=0.01,
    sphere_index=None,   # index in mesh_paths of the sphere mesh
    force_sphere_z0=True
):
    """
    Compute a single affine map for ALL meshes:
        x' = s * (x - center) + z_shift * e_z
    so that every mesh fits inside [-1,1]^3 with a small pad,
    and (optionally) the sphere's centroid ends up at z=0.

    Returns:
        s          : scalar scale
        center     : (3,) center used before scaling
        z_shift    : scalar shift applied post-scale along z (default 0)
    """
    if transforms is None:
        transforms = [np.eye(4) for _ in mesh_paths]
    assert len(transforms) == len(mesh_paths)

    mins, maxs = [], []
    centroids_world = []  # to fetch sphere centroid if needed

    for p, T in zip(mesh_paths, transforms):
        m = o3d.io.read_triangle_mesh(p)
        if len(m.vertices) == 0:
            raise ValueError(f"Empty mesh at {p}")
        m_t = o3d.geometry.TriangleMesh(m)
        if T is not None:
            m_t.transform(T)

        aabb = m_t.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.get_min_bound()))
        maxs.append(np.asarray(aabb.get_max_bound()))
        centroids_world.append(np.asarray(m_t.get_center()))

    mins = np.stack(mins, axis=0)
    maxs = np.stack(maxs, axis=0)

    min_all = mins.min(axis=0)
    max_all = maxs.max(axis=0)

    center = 0.5 * (min_all + max_all)
    extent_long = (max_all - min_all).max()
    extent_long = max(extent_long * (1.0 + pad_frac), 1e-9)

    # Map longest side → 2.0 (span of [-1,1])
    s = 2.0 / extent_long

    # z shift so the sphere ends up at z=0 (after subtracting center and scaling)
    z_shift = 0.0
    if force_sphere_z0 and sphere_index is not None:
        cz_world = centroids_world[sphere_index][2]
        z_shift = -s * (cz_world - center[2])  # bring sphere centroid's z to 0 in unit2

    return s, center, z_shift


def apply_normalizer_unit2(V_world, s, center, z_shift=0.0):
    """
    World → unit2 vertices:
        V_u2 = s * (V_world - center) + [0,0,z_shift]
    """
    V_u2 = (V_world - center[None, :]) * s
    if z_shift != 0.0:
        V_u2[:, 2] += z_shift
    return V_u2

def canonical_sdf_u2_with_shared_norm(mesh_path, s, center, R_x90, z_shift=0.0, res0=128):
    """
    Loads mesh, applies the *shared* normalizer to place it in [-1,1]^3,
    then computes canonical SDF (unit2).
    """
    m = trimesh.load(mesh_path, force='mesh', process=False)
    V_raw = m.vertices.view(np.ndarray).astype(np.float64)
    F     = m.faces.view(np.ndarray).astype(np.int32)
    V_raw = V_raw @ R_x90.T  # rotate mesh to canonical frame
    V_u2 = apply_normalizer_unit2(V_raw, s, center, z_shift)
    # small safety clamp
    V_u2 = np.clip(V_u2, -1 + 1e-6, 1 - 1e-6)

    level = 2.0 / res0
    sdf0_raw, _ = mesh2sdf.compute(V_u2, F, res0, fix=False, level=level, return_mesh=True)
    sdf0 = np.asarray(sdf0_raw).reshape(res0, res0, res0)
    return V_u2, F, sdf0


def sample_ST_no_clip_u2_xyOnly_fixed_z(
    V_u2, R, *,
    margin_vox=1, res=64,
    scale_range=(0.7, 1.0),  # usually keep 1 after shared normalization
    rng=None, slack=1e-4,
    extra_inner_margin_vox=0.0
):
    """
    Like your xyOnly sampler, but enforce tz=0 in unit2.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = float(margin_vox)/res + float(extra_inner_margin_vox)/res

    vmin = V_u2.min(axis=0); vmax = V_u2.max(axis=0)
    c0   = 0.5*(vmin + vmax)
    Vr   = (V_u2 - c0) @ R
    vminR = Vr.min(axis=0); vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)

    s_max = np.min((2.0 - 2.0*m - 2.0*slack) / widths)
    if not np.isfinite(s_max) or s_max <= 0:
        raise RuntimeError("No feasible scale with given rotation/margin.")

    s_hi = min(scale_range[1], s_max)
    s_lo = min(scale_range[0], s_hi)
    s    = rng.uniform(s_lo, s_hi)
    s    = min(s, 0.999*s_max)

    t_lo = -1.0 + m + slack - s*vminR
    t_hi =  1.0 - m - slack - s*vmaxR
    if np.any(t_lo > t_hi):
        raise RuntimeError("Empty translation range after scaling.")

    # sample (tx, ty); keep tz = 0 (but clamp into feasible z-interval just in case)
    tx = rng.uniform(t_lo[0], t_hi[0])
    ty = rng.uniform(t_lo[1], t_hi[1])
    tz = float(np.clip(0.0, t_lo[2], t_hi[2]))  # usually remains 0 thanks to z_shift + pad

    t = np.array([tx, ty, tz], dtype=np.float64)
    V_pose = ((V_u2 - c0) @ R) * s + t
    return V_pose, s, t, c0

import numpy as np

def sample_scene_xy_sphere_z0_hard(
    V_all_u2,        # np.vstack of all vertices (hand+object+sphere) in unit2
    C_s_u2,          # sphere centroid in unit2 (before pose)
    R, *,            # 3x3 rotation to apply
    res=64,
    margin_vox=1,
    scale_range=(1.0, 1.0),
    c0_scene=None,
    rng=None,
    slack=1e-4
):
    """
    Sample a single (s_aug, t) for the WHOLE scene so that:

      - geometry fits in [-1,1]^3 with the margin
      - sphere centroid lands EXACTLY at z = 0 after pose
      - t_x, t_y are chosen (here: centered in their feasible range)

    No clipping is needed on t_z; we choose s_aug so that the sphere-at-0
    constraint is compatible with the bounding box constraints.
    """
    if c0_scene is None:
        c0_scene = np.zeros(3, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    # margin in unit2
    m = float(margin_vox) / res

    # Vertices in rotated frame about shared pivot
    Vr = (V_all_u2 - c0_scene) @ R          # shape (N,3)
    vminR = Vr.min(axis=0)
    vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)

    # -------------------
    # 1) s_max from X,Y   (standard "fit in [-1,1]^3 with margin")
    # -------------------
    # side length usable along each axis: 2 - 2*m - 2*slack
    usable = 2.0 - 2.0 * m - 2.0 * slack
    if usable <= 0:
        raise RuntimeError("Margin+slack too large; no usable volume.")

    s_max_xy = np.min(usable / widths[:2])  # only x,y for tx,ty feasibility

    # -------------------
    # 2) s_max from Z with sphere forced to z=0
    # -------------------
    z_vals = Vr[:, 2]                                   # all vertices z
    z_rot  = ((C_s_u2 - c0_scene) @ R)[2]               # sphere z
    a = z_vals - z_rot                                  # offsets from sphere

    # allowable max |z'| = 1 - m - slack
    Z = 1.0 - m - slack
    if Z <= 0:
        raise RuntimeError("Margin+slack too large along z; no usable extent.")

    max_abs_a = np.max(np.abs(a))
    if max_abs_a < 1e-9:
        # all z coincide with sphere z; no extra constraint
        s_max_z = np.inf
    else:
        s_max_z = Z / max_abs_a

    # -------------------
    # 3) Combine constraints + user scale_range
    # -------------------
    s_upper = min(s_max_xy, s_max_z, scale_range[1])
    s_lower = scale_range[0]

    if not np.isfinite(s_upper) or s_upper <= 0 or s_lower > s_upper:
        raise RuntimeError(
            f"No feasible s with sphere at z=0 and margins. "
            f"s_upper={s_upper}, s_lower={s_lower}"
        )

    # Pick a scale in the feasible interval (you can choose s_upper if you
    # want maximal size instead of randomness)
    s_aug = rng.uniform(s_lower, s_upper)
    # s_aug = s_upper  # uncomment if you prefer largest possible

    # -------------------
    # 4) Compute translation ranges with this s_aug
    # -------------------
    t_lo = -1.0 + m + slack - s_aug * vminR
    t_hi =  1.0 - m - slack - s_aug * vmaxR

    if np.any(t_lo > t_hi):
        # Should not normally happen if s_aug ≤ s_max_xy & s_max_z
        raise RuntimeError("Empty translation interval even after z-aware scaling.")

    # Center in x,y (or sample if you want randomness)
    tx = 0.5 * (t_lo[0] + t_hi[0])
    ty = 0.5 * (t_lo[1] + t_hi[1])

    # -------------------
    # 5) Enforce sphere at z=0 exactly (no clip)
    # -------------------
    tz = - s_aug * z_rot

    # Optional sanity check: ensure tz lies in [t_lo[2], t_hi[2]]
    if not (t_lo[2] - 1e-6 <= tz <= t_hi[2] + 1e-6):
        # due to numerical noise we allow epsilon, but this shouldn't fail hard
        print(
            f"[WARN] tz={tz:.6f} slightly outside "
            f"[{t_lo[2]:.6f}, {t_hi[2]:.6f}] (numeric tolerance)"
        )

    return s_aug, np.array([tx, ty, tz], dtype=float)


def sample_scene_xy_sphere_z0(
    V_all_u2,        # np.vstack of all vertices (hand+object+sphere) in unit2
    C_s_u2,          # sphere centroid in unit2 (before pose)
    R, *,            # 3x3 rotation to apply
    res=64,
    margin_vox=1,
    scale_range=(1.0, 1.0),
    c0_scene=np.zeros(3, dtype=float),
    rng=None,
    slack=1e-4
):
    """
    Sample a single (s_aug, t) for the WHOLE scene so that:
      - geometry fits in [-1,1]^3 with the margin
      - t_x, t_y are random within feasible ranges
      - t_z is chosen so the SPHERE centroid lands at z=0 after pose
    """
    if rng is None:
        rng = np.random.default_rng()

    m = float(margin_vox)/res

    # Extents of the UNION after rotation about the shared pivot
    Vr = (V_all_u2 - c0_scene) @ R
    vminR = Vr.min(axis=0); vmaxR = Vr.max(axis=0)
    widths = np.maximum(vmaxR - vminR, 1e-12)

    # Feasible scale
    s_max = np.min((2.0 - 2.0*m - 2.0*slack) / widths)
    if not np.isfinite(s_max) or s_max <= 0:
        raise RuntimeError("No feasible scene scale.")
    s_hi = min(scale_range[1], s_max)
    s_lo = min(scale_range[0], s_hi)
    s_aug = min(rng.uniform(s_lo, s_hi), 0.999*s_max)

    # Feasible translation intervals (per-axis)
    t_lo = -1.0 + m + slack - s_aug * vminR
    t_hi =  1.0 - m - slack - s_aug * vmaxR
    if np.any(t_lo > t_hi):
        raise RuntimeError("Empty scene translation interval; increase margin or reduce scale.")

    # Random TX, TY within feasible box
    # tx = rng.uniform(t_lo[0], t_hi[0])
    # ty = rng.uniform(t_lo[1], t_hi[1])
    tx = 0.5 * (t_lo[0] + t_hi[0])
    ty = 0.5 * (t_lo[1] + t_hi[1])  

    # Choose TZ so the sphere centroid lands at z=0 AFTER pose
    z_rot = ( (C_s_u2 - c0_scene) @ R )[2]    # sphere center's z after rotation (pre-scale)
    t_z_target = - s_aug * z_rot               # cancels it to force z'=0
    tz = float(np.clip(t_z_target, t_lo[2], t_hi[2]))

    # If we had to clip, we can’t hit z=0 exactly; you can print a small warning if needed.
    return s_aug, np.array([tx, ty, tz], dtype=float)


def post_adjustment_sdf_mod(instance_name, root, visualize=True, save_bool=False, save_folder = None):
    print(f"Processing instance: {instance_name}")

    # Now, construct the path to the corresponding mesh file.
    # ply_mesh_path = os.path.join(root, 'renders', instance_name, "mesh.ply")

    # if not os.path.exists(ply_mesh_path):
    #     print(f"  [Warning] Corresponding mesh file not found, skipping: {ply_mesh_path}")
    #     return
    # ply_mesh_path ='/home/gcaddeo-iit.local/data_new_hand/isaac_1903.ply'
    # root_file = os.path.dirname(os.path.abspath(__file__))
    instance_separate = instance_name.rsplit('_', 1)
    instance_base = instance_separate[0]
    instance_idx = instance_separate[1]
    ply_mesh_path = os.path.join(root, "subset_google", instance_base, "isaac_"+instance_idx+".obj")
    ply_object_path = os.path.join(root, "subset_google", instance_base,"object.obj")
    ply_sphere_path = os.path.join(root, "subset_google", instance_base, "contacts_sphere_"+instance_idx+".obj")
    # print(f"  Mesh path: {ply_mesh_path}")
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
            mesh_paths = [ply_mesh_path, ply_object_path, ply_sphere_path]
            start_time = timeit.default_timer()

            # 0) Shared normalizer over hand+object+sphere
            R_x90 = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=float)

            s, center, z_shift = compute_shared_normalizer_unit2_trimesh(
                mesh_paths,
                R_x90,
                pad_frac=0.02,
                sphere_index=2,
                force_sphere_z0=True
            )
            
            Vh_u2, Fh, sdfh0 = canonical_sdf_u2_with_shared_norm_trimesh(ply_mesh_path,   s, center, R_x90, z_shift, res0=64)
            Vo_u2, Fo, sdfo0 = canonical_sdf_u2_with_shared_norm_trimesh(ply_object_path, s, center, R_x90, z_shift, res0=64)
            Vs_u2, Fs, sdfs0 = canonical_sdf_u2_with_shared_norm_trimesh(ply_sphere_path, s, center, R_x90, z_shift, res0=64)

            # 1) Scene pivot = union AABB center (like c0 but for all vertices)
            V_all = np.vstack([Vh_u2, Vo_u2, Vs_u2])
            vmin_all = V_all.min(axis=0)
            vmax_all = V_all.max(axis=0)
            c0_scene = 0.5 * (vmin_all + vmax_all)

            # sphere centroid in unit2 before pose
            C_s_u2 = Vs_u2.mean(axis=0)

            # 2) Rotation: do EXACTLY what you did before for the single mesh
            R_pose = np.array(meta["frames"][frame_id]["transform_matrix"], float)[:3, :3] 
            R_fixed=R_pose
            # 3) Sample a single (s_aug_scene, t_aug_scene) for the whole scene
            s_aug_scene, t_aug_scene = sample_scene_xy_sphere_z0_hard(
                V_all_u2=V_all,
                C_s_u2=C_s_u2,
                R=R_pose,
                res=64,
                margin_vox=1,
                scale_range=(0.5, 1.0),
                c0_scene=c0_scene
            )

            def apply_scene_ST(V_u2, R, s, t, c0):
                return ((V_u2 - c0) @ R ) * s + t

            # 4) Pose all three meshes with the SAME similarity
            Vh_pose = apply_scene_ST(Vh_u2, R_pose, s_aug_scene, t_aug_scene, c0_scene)
            Vo_pose = apply_scene_ST(Vo_u2, R_pose, s_aug_scene, t_aug_scene, c0_scene)
            Vs_pose = apply_scene_ST(Vs_u2, R_pose, s_aug_scene, t_aug_scene, c0_scene)

            # Just to check sphere centering
            C_s_after = ((C_s_u2 - c0_scene) @ R_pose) * s_aug_scene + t_aug_scene
            # print("Sphere z after pose (target ≈ 0, but may be clipped):", C_s_after[2])

            # 5) Shared grid centers
            _, Y = resample_sdf_u2(sdfo0, np.eye(3), 1.0, np.zeros(3), np.zeros(3), target_res=64)

            # 6) Resample ALL SDFs with the SAME (R_pose, s_aug_scene, t_aug_scene, c0_scene)
            phi_h, _ = resample_sdf_u2(sdfh0, R_pose, s_aug_scene, t_aug_scene, c0_scene, target_res=64, order=3, cval=4.0)
            phi_o, _ = resample_sdf_u2(sdfo0, R_pose, s_aug_scene, t_aug_scene, c0_scene, target_res=64, order=3, cval=4.0)
            phi_s, _ = resample_sdf_u2(sdfs0, R_pose, s_aug_scene, t_aug_scene, c0_scene, target_res=64, order=3, cval=4.0)

            # 7) Signs + refinement exactly as before
            inside_h = sign_from_winding_u2(Vh_pose, Fh, Y)
            inside_o = sign_from_winding_u2(Vo_pose, Fo, Y)
            inside_s = sign_from_winding_u2(Vs_pose, Fs, Y)

            dx = 2.0 / 64

            sdf_h = refine_band_exact_u2(phi_h, inside_h, Vh_pose, Fh, Y, band_vox=2, dx=dx) - 0.012
            sdf_h = skfmm.distance(np.ma.MaskedArray(sdf_h, mask=np.zeros_like(sdf_h, dtype=bool)), dx=dx)
            ss_h  = binary_fill_holes(occ_from_V_pose_u2(Vh_pose, Fh, res=64))

            sdf_o = refine_band_exact_u2(phi_o, inside_o, Vo_pose, Fo, Y, band_vox=2, dx=dx) - 0.012
            sdf_o = skfmm.distance(np.ma.MaskedArray(sdf_o, mask=np.zeros_like(sdf_o, dtype=bool)), dx=dx)
            ss_o  = binary_fill_holes(occ_from_V_pose_u2(Vo_pose, Fo, res=64))

            sdf_s = refine_band_exact_u2(phi_s, inside_s, Vs_pose, Fs, Y, band_vox=2, dx=dx) - 0.012
            sdf_s = skfmm.distance(np.ma.MaskedArray(sdf_s, mask=np.zeros_like(sdf_s, dtype=bool)), dx=dx)
            ss_s  = binary_fill_holes(occ_from_V_pose_u2(Vs_pose, Fs, res=64))


            # eps = dx/2    # tolerance ~1.5 voxels, tune as you like

            # contact_mask = (np.abs(sdf_h) <= eps) & (np.abs(sdf_o) <= eps)
            contact_mask = ss_h & ss_o   
            from scipy.ndimage import distance_transform_edt
            contact_coords = np.argwhere(contact_mask) 
            # contact_mask == True at contact → we want distance to those
            inverse = ~contact_mask           # False at contact; True elsewhere
            dist_to_contact = distance_transform_edt(inverse, sampling=dx)
            # print(dist_to_contact.min(), dist_to_contact.max(), dist_to_contact.shape)
            # V_pose_u2, s_aug, t_aug, c0 = sample_ST_no_clip_u2_xyOnly(V_u2, R_fixed, margin_vox=1, res=64, scale_range=(0.5,0.7))
            # 2) Resample canonical SDF into posed 64³
            # phi, Y = resample_sdf_u2(sdf0, R_fixed, s_aug, t_aug, c0, target_res=64, order=3, cval=4.0)

            # # 3) Get sign from posed mesh (robust)
            # inside = sign_from_winding_u2(V_pose_u2, F, Y)
            # #print(timeit.default_timer() - start_time)
            # # 4) Narrow-band refinement (keeps thin legs)
            # sdf = refine_band_exact_u2(phi, inside, V_pose_u2, F, Y, band_vox=2, dx=2.0/64)-0.012
            # #print(timeit.default_timer() - start_time)
            # dx = 2.0/64
            # sdf = skfmm.distance(np.ma.MaskedArray(sdf, mask=np.zeros_like(sdf, dtype=bool)), dx=dx)
            # # 5) (Optional) occupancy for visualization, from posed mesh in unit2
            # ss = occ_from_V_pose_u2(V_pose_u2, F, res=64) ; ss = binary_fill_holes(ss)

            # image = Image.open(os.path.join(root, 'merged_renders_cond3','renders_cond', instance_name, f"{frame_id:03d}.png")).convert('RGBA')
            # cropped_rgb, dbg = crop_image_pose_faithful(
            #     image, V_u2, V_pose_u2,
            #     c2w=np.array(meta["frames"][frame_id]["transform_matrix"], dtype=float),
            #     cam_angle_x=float(meta["frames"][frame_id]["camera_angle_x"]),
            #     out_size=518, pad_ratio=1.15, min_crop_px=8
            # # )
            # cropped_rgb = collage_from_grid_bbox_preserve_aspect_negz(
            #     image, occ=ss, out_size=518
            # )

            # # print("DBG:", dbg)
            # cropped_rgb.save(f"{instance_name}_f{frame_id:03d}_cropped.png")

            if save_bool:
                paths = save_scene_sdfs_only_correct(
                    out_dir    = os.path.join(save_folder, instance_name),
                    scene_name       = f"{instance_name}_f{frame_id:03d}",
                    items     = {
                        "hand":   {"sdf": sdf_h, "occ": ss_h},   # or {"sdf": sdf_h, "occ": None} to derive idx from sdf<=0
                        "object": {"sdf": sdf_o, "occ": ss_o},
                        # "sphere": {"sdf": sdf_s, "occ": ss_s},  # <-- simply don't include it
                    },                               # float32 [64,64,64]
                    res        = 64,
                    R_fixed    = R_fixed,                           # 3x3 rotation used for pose
                    s_aug      = s_aug_scene,                             # sampled scale
                    t_aug      = t_aug_scene,                             # sampled translation
                    c0         = c0_scene,                                # center used for about-center transform
                    frame_id   = frame_id,
                    margin_vox = 1,
                    contact_mask=contact_mask,
                    dist_to_contact=dist_to_contact,
                    dx = dx
                )
                print(f"Saved {instance_name}_f{frame_id:03d}")
                # idx_hand   = np.load(os.path.join(save_folder, instance_name, "idxs", f"{instance_name}_f{frame_id:03d}__hand.npy"))
                # idx_object = np.load(os.path.join(save_folder, instance_name, "idxs", f"{instance_name}_f{frame_id:03d}__object.npy"))

                # # shared grid res from the single meta
                # with open(os.path.join(save_folder, instance_name, f"{instance_name}_f{frame_id:03d}_meta.json")) as f:
                #     meta_img = json.load(f)
                # res = int(meta_img["grid"]["res"])

                # # one RGBA render for the scene (hand + object together)
                

                # collaged = collage_from_union_indices(
                #     image_rgba=image,
                #     idx_list=[idx_hand, idx_object],  # union bbox
                #     res=res,
                #     out_size=518,
                #     pad_ratio=0.1
                # )
                # collaged.save(os.path.join(save_folder, instance_name, f"{instance_name}_f{frame_id:03d}__hand_object_collage.png"))
                # with open(os.path.join(save_folder, instance_name, f"{instance_name}_f{frame_id:03d}_meta.json")) as f:
                #     meta_data = json.load(f)
                # # pose2d_meta = meta_data.get("pose2d_meta", None)
                # new_image = collage_from_meta_bbox_preserve_aspect(
                #     image, pose2d_meta, out_size=518
                # )
                # new_image.save(f"{instance_name}_f{frame_id:03d}_new.png")
            triples = [
                ("hand",   sdf_h,   ss_h),
                ("object", sdf_o, ss_o),
                ("sphere", sdf_s, ss_s),
            ]
            colors = {"hand":"green", "object":"orange", "sphere":"cyan"}  # tweak as you like

            if visualize:

                if save_bool:
                    print(f"  SDF loaded from: {sdf_folder}")
                    triples[0]= ("hand", np.load(os.path.join(save_folder, instance_name, "sdfs", f"{instance_name}_f{frame_id:03d}__hand.npy")), ss_h)
                    triples[1]= ("object",np.load(os.path.join(save_folder, instance_name, "sdfs", f"{instance_name}_f{frame_id:03d}__object.npy")), ss_o)
                    # ss = np.load(os.path.join(save_folder, instance_name, "occs", f"{instance_name}_f{frame_id:03d}.npy"))
                # sdf = shift(sdf_tensor, shift_in_grid.numpy())
                # === Data preparation for plotting ===
                threshold = 0.005

                # ---------------- Static plot: voxel shells + |SDF|<=threshold points ----------------
                print("\n--- Displaying Static Plot (3 meshes) ---")
                static_plotter = pv.Plotter(window_size=[1000, 800])
                def make_point_grid_from_bool(occ):
                    g = pv.ImageData()
                    g.dimensions = np.array(occ.shape)  # using point_data like you already do
                    g.point_data['occ'] = occ.astype(np.uint8).ravel(order='F')
                    return g

                def make_point_grid_from_sdf(sdf):
                    g = pv.ImageData()
                    g.dimensions = np.array(sdf.shape)
                    g.point_data['sdf'] = sdf.ravel(order='F')
                    return g
                for name, sdf_i, occ_i in triples:
                    # voxel shell
                    g_occ = make_point_grid_from_bool(occ_i)
                    shell = g_occ.contour([0.5], scalars='occ')
                    static_plotter.add_mesh(shell, color=colors[name], opacity=0.35, label=f"{name} shell")

                # union of near-zero SDF points (or do per-mesh if you prefer)
                union_mask = np.zeros_like(triples[0][1], dtype=bool)
                for _, sdf_i, _ in triples:
                    union_mask |= (np.abs(sdf_i) <= threshold)

                coords = np.argwhere(union_mask)
                if coords.size:
                    static_plotter.add_points(pv.PolyData(coords), color="purple",
                                            opacity=0.8, point_size=7, render_points_as_spheres=True,
                                            label=f"|φ| ≤ {threshold} (union)")
                if contact_coords.size:
                    static_plotter.add_points(
                        pv.PolyData(contact_coords),
                        color="red",
                        opacity=1.0,
                        point_size=10,
                        render_points_as_spheres=True,
                        label="hand–object contacts",
                    )

                static_plotter.add_axes(); static_plotter.show_grid(); static_plotter.add_legend()
                static_plotter.show(title=f"Static SDF vs Voxel Shell (all)")

                # ---------------- Plot 1: Interactive volumetric |SDF| region (union) ----------------
                print("\n--- Displaying Interactive Plot 1 (3 meshes): Volumetric |SDF| Region ---")
                p1 = pv.Plotter(window_size=[1000, 800])
                for name, _, occ_i in triples:
                    g_occ = make_point_grid_from_bool(occ_i)
                    shell = g_occ.contour([0.5], scalars='occ')
                    p1.add_mesh(shell, color=colors[name], opacity=0.25, label=f"{name} shell")
                p1.add_text("Voxel Shells + Interactive |SDF| Region (union)")

                def callback_abs_sdf_volume(x_value):
                    p1.remove_actor("sdf_point_cloud", render=False)
                    dyn_thr = x_value + threshold
                    if dyn_thr < 0:
                        return
                    mask = np.zeros_like(triples[0][1], dtype=bool)
                    for _, sdf_i, _ in triples:
                        mask |= (np.abs(sdf_i) <= dyn_thr)
                    coords = np.argwhere(mask)
                    if coords.size:
                        p1.add_points(pv.PolyData(coords), name="sdf_point_cloud",
                                    color="purple", opacity=0.8, point_size=7, render_points_as_spheres=True)

                p1.add_slider_widget(callback=callback_abs_sdf_volume, rng=[-0.5, 0.5], value=0.0,
                                    title="x, where |φ| ≤ x + threshold", style='modern')
                p1.add_axes(); p1.show_grid(); p1.add_legend()
                p1.show(title="Interactive |SDF| Region (union)")

                # ---------------- Plot 2: Interactive SDF level sets (each mesh) ----------------
                print("\n--- Displaying Interactive Plot 2 (3 meshes): SDF Level Set ---")
                p2 = pv.Plotter(window_size=[1000, 800])
                p2.add_text("Interactive SDF isosurface φ = t (per mesh)")

                # create a grid per mesh once
                sdf_grids = []
                for name, sdf_i, _ in triples:
                    sdf_grids.append((name, make_point_grid_from_sdf(sdf_i)))
                if contact_coords.size:
                    p2.add_points(
                        pv.PolyData(contact_coords),
                        name="contacts",
                        color="red",
                        opacity=1.0,
                        point_size=11,
                        render_points_as_spheres=True,
                        label="hand–object contacts",
                    )

                def callback_single_sdf(threshold2):
                    # remove previous iso surfaces
                    for name, _ in sdf_grids:
                        p2.remove_actor(f"iso_{name}", render=False)
                    # add new level for each mesh
                    for name, g_sdf in sdf_grids:
                        iso = g_sdf.contour([threshold2], scalars='sdf')
                        if "sphere" in name:
                            opacity =1
                        else:
                            opacity =0.2
                        # optional decimation for speed:
                        # iso = iso.decimate(0.5)
                        p2.add_mesh(iso, name=f"iso_{name}", color=colors[name], opacity=opacity, label=f"{name} φ={threshold2:.3f}")

                p2.add_slider_widget(callback=callback_single_sdf, rng=[-1.0, 1.0], value=0.0,
                                    title="φ isovalue", style='modern')
                p2.add_axes(); p2.show_grid(); p2.add_legend()
                p2.show(title="Interactive SDF Isosurface (hand+object+sphere)")

                # ---------------- Plot 3: Interactive Y-slice with binned contours (union field) ----------------
                print("\n--- Displaying Plot 3 (3 meshes): SDF Slice (union) ---")
                # union SDF = min of φ_i  (standard R-function for union)
                sdf_union = np.minimum.reduce([sdf_i for _, sdf_i, _ in triples])

                sdf_min, sdf_max = sdf_union.min(), sdf_union.max()
                num_bins = 10
                delta = sdf_max - sdf_min
                epsilon = 1e-4 * max(delta, 1.0)
                log_spaced = np.logspace(np.log10(epsilon), np.log10(delta + epsilon), num_bins + 1)
                contour_levels = sdf_min + log_spaced - epsilon
                contour_levels[0] = sdf_min
                contour_levels[-1] = sdf_max

                g_union = make_point_grid_from_sdf(sdf_union)

                p3 = pv.Plotter(window_size=[1000, 800])
                p3.set_background('darkgrey')
                # def update_slice_y(y_pos):
                #     p3.remove_actor("slice_mesh",  render=False)
                #     p3.remove_actor("slice_edges", render=False)
                #     slice_y = g_union.slice(normal='y', origin=(g_union.center[0], y_pos, g_union.center[2]))
                #     if slice_y.n_points == 0:
                #         return
                #     p3.add_mesh(slice_y, name="slice_mesh", scalars='sdf', cmap='viridis',
                #                 n_colors=num_bins, clim=[sdf_min, sdf_max], show_scalar_bar=False)
                #     edges = slice_y.contour(contour_levels, scalars='sdf')
                #     p3.add_mesh(edges, name="slice_edges", color='white', line_width=1, opacity=0.4)
                def update_slice_y(y_pos):
                    p3.remove_actor("slice_mesh",  render=False)
                    p3.remove_actor("slice_edges", render=False)
                    p3.remove_actor("contact_slice", render=False)  # <-- new

                    slice_y = g_union.slice(normal='y', origin=(g_union.center[0], y_pos, g_union.center[2]))
                    if slice_y.n_points == 0:
                        return

                    p3.add_mesh(
                        slice_y,
                        name="slice_mesh",
                        scalars='sdf',
                        cmap='viridis',
                        n_colors=num_bins,
                        clim=[sdf_min, sdf_max],
                        show_scalar_bar=False,
                    )
                    edges = slice_y.contour(contour_levels, scalars='sdf')
                    p3.add_mesh(edges, name="slice_edges", color='white', line_width=1, opacity=0.4)

                    # ---- ADD: contact voxels in this Y slice ----
                    # g_union is ImageData with default origin=(0,0,0), spacing=(1,1,1)
                    # so y_pos ≈ j index
                    j_idx = int(round(y_pos))
                    if 0 <= j_idx < contact_mask.shape[1]:
                        contact_slice = contact_mask[:, j_idx, :]   # (x,z)
                        cs_coords_xz = np.argwhere(contact_slice)   # (M,2) (i,k)
                        if cs_coords_xz.size > 0:
                            # Build full (i,j,k) coords
                            pts = np.zeros((cs_coords_xz.shape[0], 3), dtype=float)
                            pts[:, 0] = cs_coords_xz[:, 0]
                            pts[:, 1] = j_idx
                            pts[:, 2] = cs_coords_xz[:, 1]
                            p3.add_points(
                                pv.PolyData(pts),
                                name="contact_slice",
                                color="red",
                                opacity=1.0,
                                point_size=9,
                                render_points_as_spheres=True,
                            )

                y0 = g_union.center[1]
                update_slice_y(y0)
                p3.add_scalar_bar(title='SDF (union)')
                y_bounds = g_union.bounds[2:4]
                p3.add_slider_widget(callback=update_slice_y, rng=y_bounds, value=y0,
                                    title="Y Slice Position", style='modern')
                p3.add_text("Interactive SDF Slice (union of 3 meshes)")
                p3.view_xz()
                p3.show(title="Interactive SDF Slice (union)")


                print("\n--- Displaying Plot 4: Distance to hand–object contacts ---")

                g_dist = make_point_grid_from_sdf(dist_to_contact)  # same helper as for SDF

                d_min, d_max = float(dist_to_contact.min()), float(dist_to_contact.max())

                p4 = pv.Plotter(window_size=[1000, 800])
                p4.add_text("Isosurfaces of distance-to-contact (same units as SDF)")

                # optional: show voxel shells for context
                for name, _, occ_i in triples:
                    g_occ = make_point_grid_from_bool(occ_i)
                    shell = g_occ.contour([0.5], scalars='occ')
                    p4.add_mesh(shell, color=colors[name], opacity=0.15)

                # also show contact points themselves
                if contact_coords.size:
                    p4.add_points(
                        pv.PolyData(contact_coords),
                        name="contacts",
                        color="red",
                        opacity=1.0,
                        point_size=11,
                        render_points_as_spheres=True,
                        label="contacts",
                    )

                def callback_dist_iso(d_thr):
                    p4.remove_actor("dist_iso", render=False)
                    iso = g_dist.contour([d_thr], scalars='sdf')  # here 'sdf' is actually distance
                    p4.add_mesh(iso, name="dist_iso", color="yellow", opacity=0.7)

                # choose a useful range, e.g. from nearly 0 to some fraction of the box
                p4.add_slider_widget(
                    callback=callback_dist_iso,
                    rng=[0.0, min(0.3, d_max)],
                    value=0.05,
                    title="distance to contact (same units as SDF)",
                    style='modern',
                )

                p4.add_axes(); p4.show_grid()
                p4.show(title="Distance to hand–object contacts")
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]  # last entry
            print(f"  [ERROR] Failed to process {instance_name}: {e} (line {tb.lineno})")


SKIP = {}

NPROC         = 16
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
            save_folder=os.path.join(root, "data_pose_norm_grasps"),
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
    parser.add_argument("--root", type=str, default='/home/user/TRELLIS/datasets/Hands_Google', help="Root directory for instances.")

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
    #     # if instance_name =='2e35f84025d83fcd1d1eb082bbbd4dc8b991ab501d81eae7a8a710012a58e38a' or instance_name== '118519d1320c7f18f2fad814d8d05337ac12f6d10e9e40946c40b60c94d36b25':
    #     #    continue
    #     post_adjustment_sdf_mod(instance_name, args.root, save_bool=True, save_folder=os.path.join(args.root,'data_pose_norm'), visualize=False)




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Superimpose SDF and voxel grid.")
#     parser.add_argument("--resolution", type=int, default=64, help="Resolution of the SDF grid.")
#     parser.add_argument("--min_efficiency", type=float, default=0.15, help="Minimum bounding box efficiency for mesh density.")
#     parser.add_argument("--pitch", type=float, default=0.01, help="Pitch for bounding box efficiency calculation.")
#     parser.add_argument("--root", type=str, default='/home/gcaddeo-iit.local/robot-code/3dreconstruction', help="Root directory for instances.")

#     args = parser.parse_args()

#     resolution = args.resolution
#     print(f"Using resolution: {resolution}")
#     instances = glob.glob(os.path.join(args.root, "merged_renders_cond3/renders_cond") + "/*")


#     for instance in instances:
#         instance_name = os.path.basename(instance)
#         post_adjustment_sdf_mod(instance_name, args.root, save_bool=True, save_folder=os.path.join(args.root,'data'), visualize=True)
