"""
Standard paired mesh metrics -- CD, NC, F@tau, EMD -- for the a/b guidance eval.

Formulas are copied from /projects/gcaddeo/inference/TRELLIS/eval_meshes_paired_emd_voxel.py
(chamfer_from_dists, normal_consistency, fscore_from_dists, emd_l2) so numbers are
directly comparable to previous evaluations. Two deliberate simplifications:

  - No ICP / normalization: here both meshes come from the SAME 64^3 SDF grid in the
    same unit frame ([-1,1], 1 voxel = 2/64), so they are already aligned. The
    inference-repo script needed multi-init ICP because its recon and GT meshes came
    from different frames.
  - Point-to-point distances via cKDTree (the script's fallback mode), not
    point-to-triangle: at 10k samples per surface the difference is far below the
    arm-to-arm effects being measured.

Voxel-IoU is intentionally NOT computed here: the harness computes it exactly from
SDF occupancy (occ_iou_gt), which is what mesh re-voxelization approximates.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from skimage.measure import marching_cubes


def sdf_to_mesh(sdf: np.ndarray) -> trimesh.Trimesh | None:
    """64^3 SDF (unit cube [-1,1], voxel-center samples) -> trimesh, or None if no surface.

    Same convention as inference_dex.py's save_mesh: transpose to ZYX for skimage,
    convert back, shift by half a voxel because samples live at voxel centers.
    """
    if sdf.min() > 0 or sdf.max() < 0:
        return None
    dx = 2.0 / sdf.shape[0]
    vol_zyx = np.transpose(sdf, (2, 1, 0))
    try:
        verts_zyx, faces, normals_zyx, _ = marching_cubes(vol_zyx, level=0.0,
                                                          spacing=(dx, dx, dx))
    except (ValueError, RuntimeError):
        return None
    verts = verts_zyx[:, [2, 1, 0]] + np.array([-1.0, -1.0, -1.0]) + 0.5 * dx
    normals = normals_zyx[:, [2, 1, 0]]
    return trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)


def _sample_surface(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(2**31)))
    nrm = mesh.face_normals[face_idx]
    return np.asarray(pts, dtype=np.float64), np.asarray(nrm, dtype=np.float64)


# ---- formulas below match eval_meshes_paired_emd_voxel.py -------------------------

def chamfer_from_dists(d_ab, d_ba, squared=False):
    if squared:
        return float((d_ab * d_ab).mean() + (d_ba * d_ba).mean())
    return float(d_ab.mean() + d_ba.mean())


def normal_consistency(src_n, dst_n_at_nn):
    dots = np.sum(src_n * dst_n_at_nn, axis=1)
    return float(np.abs(dots).mean())


def fscore_from_dists(d_rec_to_gt, d_gt_to_rec, tau):
    precision = float((d_rec_to_gt < tau).mean())
    recall = float((d_gt_to_rec < tau).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def emd_l2(pts_a, pts_b, rng, n_points=1024, squared=False):
    def sub(p):
        if len(p) <= n_points:
            return p
        return p[rng.choice(len(p), size=n_points, replace=False)]
    A = sub(pts_a).astype(np.float64)
    B = sub(pts_b).astype(np.float64)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    if n == 0:
        return 0.0
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    if squared:
        C = C * C
    row, col = linear_sum_assignment(C)
    return float(C[row, col].mean())


# ------------------------------------------------------------------------------------

def paired_mesh_metrics(sdf_rec: np.ndarray, sdf_gt: np.ndarray,
                        rng: np.random.Generator,
                        n_points: int = 10000, emd_points: int = 1024,
                        tau: float = 0.02) -> dict | None:
    """CD / NC / F@tau / EMD between the meshes of two SDF grids.

    Returns None if either SDF has no zero crossing (metrics undefined).
    """
    mesh_rec = sdf_to_mesh(sdf_rec)
    mesh_gt = sdf_to_mesh(sdf_gt)
    if mesh_rec is None or mesh_gt is None:
        return None

    rec_pts, rec_nrm = _sample_surface(mesh_rec, n_points, rng)
    gt_pts, gt_nrm = _sample_surface(mesh_gt, n_points, rng)

    tree_gt = cKDTree(gt_pts)
    tree_rec = cKDTree(rec_pts)
    d_rec_to_gt, nn_gt = tree_gt.query(rec_pts)
    d_gt_to_rec, nn_rec = tree_rec.query(gt_pts)

    return {
        "cd": chamfer_from_dists(d_rec_to_gt, d_gt_to_rec, squared=False),
        "nc": 0.5 * (normal_consistency(rec_nrm, gt_nrm[nn_gt])
                     + normal_consistency(gt_nrm, rec_nrm[nn_rec])),
        f"f@{tau:g}": fscore_from_dists(d_rec_to_gt, d_gt_to_rec, tau=tau),
        "emd": emd_l2(rec_pts, gt_pts, rng, n_points=emd_points),
    }
