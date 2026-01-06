"""
check_feature_projection_single_object.py

Goal
-----
For ONE object + ONE chosen view:
1) Project voxels from the **vanilla voxelization** (mesh->open3d voxelgrid in unit2)
2) Project voxels from a **posed voxelization** (idxs/occs produced by your SDF pipeline, also in unit2)
3) Compare:
   - visual overlay on the rendered RGBA
   - alpha-coverage score (how many projected points land on the object silhouette)
   - UV invariance sanity check (transform points + compensate extrinsics)
   - DINOv2 feature sampling consistency:
        vanilla vs posed (after mapping posed points back to vanilla with the inverse pose)

This script assumes:
- Blender renders in unit2 (your updated blender script)
- transforms.json exists in:  <root>/renders/<sha256>/transforms.json
- images in:                 <root>/renders/<sha256>/<frame["file_path"]>
- mesh.ply in:               <root>/renders/<sha256>/mesh.ply
- vanilla voxel ply in:      <root>/voxels_unit2/<sha256>.ply   (or it can generate it)
- posed voxelization:
    idxs.npy (N,3) in grid indices (i,j,k) and meta json containing pose:
      meta["pose"]["R_fixed"], meta["pose"]["s_aug"], meta["pose"]["t_aug"], meta["pose"]["c0"]
    and grid origin/voxel_size (unit2):
      meta["grid"]["origin"], meta["grid"]["voxel_size"]

You can point to idxs/meta explicitly via args.

Run example
-----------
python check_feature_projection_single_object.py \
  --root /path/to/output_dir \
  --sha256 <SHA> \
  --view_id 0 \
  --posed_idxs /path/to/<name>.npy \
  --posed_meta /path/to/<name>_meta.json \
  --dinov2_model dinov2_vitl14_reg

"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import open3d as o3d

# your project deps
import utils3d


# ----------------------------
#  Pose transform helpers
# ----------------------------
def T_old_to_pose_from_row(R_row, s, t, c0, device=None, dtype=torch.float32):
    """
    Your pipeline applies (ROW-VECTOR convention):
      v_pose_row = (v_row - c0_row) @ R_row * s + t_row

    Equivalent COLUMN-VECTOR transform:
      v_pose = A @ v_old + b
      A = s * R_row^T
      b = t - A @ c0
    """
    R_row = torch.as_tensor(R_row, dtype=dtype, device=device)
    c0    = torch.as_tensor(c0,    dtype=dtype, device=device)
    t     = torch.as_tensor(t,     dtype=dtype, device=device)
    s     = torch.as_tensor(s,     dtype=dtype, device=device)

    A = s * R_row.T
    b = t - A @ c0

    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = A
    T[:3, 3]  = b
    return T


def T_pose_to_old_from_row(R_row, s, t, c0, device=None, dtype=torch.float32):
    """
    Inverse of T_old_to_pose_from_row.
    A = s * R^T  =>  A^{-1} = (1/s) * R
    """
    R_row = torch.as_tensor(R_row, dtype=dtype, device=device)
    c0    = torch.as_tensor(c0,    dtype=dtype, device=device)
    t     = torch.as_tensor(t,     dtype=dtype, device=device)
    s     = torch.as_tensor(s,     dtype=dtype, device=device)

    Ainv = (1.0 / s) * R_row
    binv = c0 - Ainv @ t

    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = Ainv
    T[:3, 3]  = binv
    return T


def apply_T(points_xyz, T4x4):
    """points_xyz: [N,3], T4x4: [4,4] -> [N,3]"""
    N = points_xyz.shape[0]
    ones = torch.ones((N, 1), dtype=points_xyz.dtype, device=points_xyz.device)
    Ph = torch.cat([points_xyz, ones], dim=1)      # [N,4]
    Qh = (T4x4 @ Ph.T).T                           # [N,4]
    return Qh[:, :3]


# ----------------------------
#  Projection + visualization
# ----------------------------
def load_rgba_518(image_path):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((518, 518), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    a   = arr[..., 3:4]
    rgb = rgb * a
    return img, rgb, a[..., 0]


def overlay_points_on_rgba(image_rgba_518, uv01, title="", max_points=20000, out_path=None):
    img = image_rgba_518
    W, H = img.size

    uv = uv01.detach().cpu().numpy()
    uv = uv[np.isfinite(uv).all(axis=1)]
    if uv.shape[0] == 0:
        print("[overlay] no finite UV points.")
        return

    if uv.shape[0] > max_points:
        idx = np.random.choice(uv.shape[0], size=max_points, replace=False)
        uv = uv[idx]

    xs = uv[:, 0] * (W - 1)
    ys = uv[:, 1] * (H - 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.asarray(img))
    plt.scatter(xs, ys, s=1, alpha=0.6)
    plt.title(title)
    plt.axis("off")
    if out_path is not None:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
    # plt.show()


def inside_alpha_ratio(image_rgba_518, uv01, alpha_thresh=10):
    img = image_rgba_518
    arr = np.asarray(img)
    alpha = arr[..., 3]   # 0..255
    H, W = alpha.shape

    uv = uv01.detach().cpu().numpy()
    uv = uv[np.isfinite(uv).all(axis=1)]
    if uv.shape[0] == 0:
        return 0.0

    xs = np.clip(np.rint(uv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    ys = np.clip(np.rint(uv[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    inside = alpha[ys, xs] > alpha_thresh
    return float(inside.mean())


def project_uv01(points_xyz, extrinsics_4x4, intrinsics_3x3):
    # utils3d.torch.project_cv returns uv in [0,1]
    uv01 = utils3d.torch.project_cv(points_xyz, extrinsics_4x4, intrinsics_3x3)[0]
    return uv01


# ----------------------------
#  Vanilla voxelization (unit2)
# ----------------------------
def voxelize_mesh_unit2(mesh_ply_path, res=64):
    mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
    vertices = np.clip(np.asarray(mesh.vertices), -1.0 + 1e-6, 1.0 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=2.0 / res,
        min_bound=(-1.0, -1.0, -1.0),
        max_bound=( 1.0,  1.0,  1.0),
    )

    ijk = np.array([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.int32)
    assert np.all(ijk >= 0) and np.all(ijk < res)

    dx = 2.0 / res
    centers = -1.0 + (ijk + 0.5) * dx
    return centers.astype(np.float32)


def read_ply_points(path):
    # utils3d.io.read_ply returns (points, colors?) usually
    pts = utils3d.io.read_ply(path)[0]
    return pts.astype(np.float32)


# ----------------------------
#  DINOv2 patchtoken sampling
# ----------------------------
def load_dinov2(model_name, device):
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval().to(device)
    return model


def preprocess_rgb518_to_tensor(rgb518):
    """
    rgb518: float32 [H,W,3] in [0,1], already premultiplied by alpha
    Returns: torch [1,3,518,518] normalized like your pipeline
    """
    x = torch.from_numpy(rgb518).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


@torch.no_grad()
def dinov2_patchtokens(model, x_1x3x518x518):
    """
    Returns patchtokens as a 2D feature map:
      patchtokens_map: [1, C, n_patch, n_patch]
    Uses the same logic you had in your extraction script.
    """
    feats = model(x_1x3x518x518.to(next(model.parameters()).device), is_training=True)
    # tokens after [CLS] and register tokens
    tokens = feats['x_prenorm'][:, model.num_register_tokens + 1:]   # [1, n_patch^2, C]
    C = tokens.shape[-1]
    n_patch = 518 // 14
    patchtokens_map = tokens.permute(0, 2, 1).reshape(1, C, n_patch, n_patch)
    return patchtokens_map


@torch.no_grad()
def sample_patchtokens(patchtokens_map, uv01):
    """
    patchtokens_map: [1, C, Hp, Wp]
    uv01: [N,2] in [0,1], image coords
    Returns: [N, C]
    """
    device = patchtokens_map.device
    uv = uv01.to(device)
    grid = uv.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0    # [1, N, 1, 2], grid_sample expects [-1,1]
    out = F.grid_sample(
        patchtokens_map, grid,
        mode='bilinear',
        align_corners=False
    )                                                   # [1, C, N, 1]
    out = out.squeeze(0).squeeze(-1).permute(1, 0)     # [N, C]
    return out


def cosine_sim(a, b, eps=1e-8):
    # a,b: [N,C]
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1)


def T_raw_to_norm(s_norm, c_norm, device=None, dtype=torch.float32):
    s = torch.tensor(s_norm, dtype=dtype, device=device)
    c = torch.tensor(c_norm, dtype=dtype, device=device)
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = torch.eye(3, dtype=dtype, device=device) * s
    T[:3, 3]  = -s * c
    return T

def T_norm_to_raw(s_norm, c_norm, device=None, dtype=torch.float32):
    s = torch.tensor(s_norm, dtype=dtype, device=device)
    c = torch.tensor(c_norm, dtype=dtype, device=device)
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = torch.eye(3, dtype=dtype, device=device) / s
    T[:3, 3]  = c
    return T

# ----------------------------
#  Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--sha256", type=str, required=True)
    parser.add_argument("--view_id", type=int, default=0)
    parser.add_argument("--res", type=int, default=64)

    # posed voxelization
    parser.add_argument("--posed_idxs", type=str, required=True, help="idxs .npy (N,3) grid indices")
    parser.add_argument("--posed_meta", type=str, required=True, help="meta json saved alongside idxs")

    # vanilla voxelization
    parser.add_argument("--vanilla_ply", type=str, default=None, help="optional path to voxels_unit2/<sha>.ply")
    parser.add_argument("--generate_vanilla_if_missing", action="store_true")

    # feature extraction
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--no_dino", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ----------------------------
    # Load transforms.json + choose one view
    # ----------------------------
    transforms_path = os.path.join(args.root, "renders_cond", args.sha256, "transforms.json")
    meta_render = json.load(open(transforms_path, "r"))
    frames = meta_render["frames"]
    assert 0 <= args.view_id < len(frames), f"view_id out of range (0..{len(frames)-1})"
    frame = frames[args.view_id]

    image_path = os.path.join(args.root, "renders_cond", args.sha256, frame["file_path"])
    assert os.path.exists(image_path), image_path

    # camera matrices (your convention)
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
    c2w[:3, 1:3] *= -1
    E_old = torch.inverse(c2w)  # extrinsics world->cam

    fov = torch.tensor(frame["camera_angle_x"], dtype=torch.float32, device=device)
    K = utils3d.torch.intrinsics_from_fov_xy(fov, fov).to(device)

    # load image
    img_rgba_518, rgb518, alpha518 = load_rgba_518(image_path)

    # ----------------------------
    # Load / build vanilla voxel centers (unit2)
    # ----------------------------
    if args.vanilla_ply is not None:
        vanilla_ply = args.vanilla_ply
    else:
        vanilla_ply = os.path.join(args.root, "voxels_unit2", f"{args.sha256}.ply")

    if os.path.exists(vanilla_ply):
        centers_old = read_ply_points(vanilla_ply)
        print(f"[vanilla] loaded ply: {vanilla_ply}  N={centers_old.shape[0]}")
    else:
        if not args.generate_vanilla_if_missing:
            raise FileNotFoundError(f"Vanilla ply not found: {vanilla_ply} (pass --generate_vanilla_if_missing)")
        mesh_ply = os.path.join(args.root, "renders", args.sha256, "mesh.ply")
        centers_old = voxelize_mesh_unit2(mesh_ply, res=args.res)
        os.makedirs(os.path.dirname(vanilla_ply), exist_ok=True)
        utils3d.io.write_ply(vanilla_ply, centers_old)
        print(f"[vanilla] generated + wrote: {vanilla_ply}  N={centers_old.shape[0]}")

    centers_old_t = torch.from_numpy(centers_old).to(device)

    # ----------------------------
    # Load posed voxelization centers from idxs + meta (unit2)
    # ----------------------------
    idxs = np.load(args.posed_idxs).astype(np.int32)  # (N,3), indices i,j,k
    posed_meta = json.load(open(args.posed_meta, "r"))
    s_norm = posed_meta["canonical"]["s_norm"]
    c_norm = posed_meta["canonical"]["c_norm"]

    T_r2n = T_raw_to_norm(s_norm, c_norm, device=device)
    T_n2r = T_norm_to_raw(s_norm, c_norm, device=device)

    origin = np.array(posed_meta["grid"]["origin"], dtype=np.float64)          # should be [-1,-1,-1]
    voxel_size = float(posed_meta["grid"]["voxel_size"])                       # should be 2/res
    centers_pose = origin + (idxs.astype(np.float64) + 0.5) * voxel_size
    centers_pose = centers_pose.astype(np.float32)
    centers_pose_t = torch.from_numpy(centers_pose).to(device)

    pose = posed_meta["pose"]
    R_row = np.array(pose["R_fixed"], dtype=np.float64)
    s     = float(pose["s_aug"])
    t     = np.array(pose["t_aug"], dtype=np.float64)
    c0    = np.array(pose["c0"],    dtype=np.float64)

    # Build transforms for "old<->pose" frames
    T_norm_to_pose = T_old_to_pose_from_row(R_row, s, t, c0, device=device)
    T_pose_to_norm = T_pose_to_old_from_row(R_row, s, t, c0, device=device)

    T_old_to_pose = T_norm_to_pose @ T_r2n
    T_pose_to_old = T_n2r @ T_pose_to_norm
    # Compensated extrinsics for posed points:
    #   X_cam = E_old * X_old, and X_old = T_pose_to_old * X_pose  => E_pose = E_old * T_pose_to_old
    E_pose = E_old @ T_pose_to_old

    print(f"[posed] idxs: {args.posed_idxs}  N={centers_pose.shape[0]}")
    print(f"[posed] meta: {args.posed_meta}")
    print(f"[pose params] s={s:.4f}, t={t}, c0={c0}")

    # ----------------------------
    # 1) Visual overlays + alpha scores
    # ----------------------------
    uv_old = project_uv01(centers_old_t,  E_old,  K)    # vanilla points + vanilla extrinsics
    uv_pos = project_uv01(centers_pose_t, E_pose, K)    # posed points + compensated extrinsics

    score_old = inside_alpha_ratio(img_rgba_518, uv_old)
    score_pos = inside_alpha_ratio(img_rgba_518, uv_pos)

    print(f"[alpha score] vanilla voxels on silhouette: {score_old:.4f}")
    print(f"[alpha score] posed    voxels on silhouette: {score_pos:.4f}")

    overlay_points_on_rgba(img_rgba_518, uv_old, title=f"Vanilla voxels overlay (view {args.view_id})", out_path='fig1.png')
    overlay_points_on_rgba(img_rgba_518, uv_pos, title=f"Pोजed voxels overlay (compensated extrinsics) (view {args.view_id})", out_path='fig2.png')

    # ----------------------------
    # 2) UV invariance sanity check (very important)
    #    Transform vanilla points into pose frame, and project with E_pose.
    #    Must match uv_old.
    # ----------------------------
    centers_old_as_pose = apply_T(centers_old_t, T_old_to_pose)
    uv_old_as_pose = project_uv01(centers_old_as_pose, E_pose, K)

    max_uv_err = (uv_old - uv_old_as_pose).abs().max().item()
    mean_uv_err = (uv_old - uv_old_as_pose).abs().mean().item()
    print(f"[UV invariance] max |uv_old - uv(transform(old))| = {max_uv_err:.6e}")
    print(f"[UV invariance] mean|uv_old - uv(transform(old))| = {mean_uv_err:.6e}")
    if max_uv_err > 1e-3:
        print("⚠️  UV invariance error is large. This usually means pose convention mismatch (R/s/t/c0 or row/col).")

    # ----------------------------
    # 3) DINO feature consistency (optional)
    # ----------------------------
    if args.no_dino:
        print("[DINO] skipped (--no_dino). Done.")
        return

    dinov2 = load_dinov2(args.dinov2_model, device=device)
    x = preprocess_rgb518_to_tensor(rgb518).to(device)

    patchtokens_map = dinov2_patchtokens(dinov2, x)  # [1,C,Hp,Wp]

    # sample features for vanilla points
    feat_old = sample_patchtokens(patchtokens_map, uv_old)  # [N_old, C]

    # sample features for posed points
    feat_pos = sample_patchtokens(patchtokens_map, uv_pos)  # [N_pos, C]

    # To compare vanilla vs posed features, we need correspondence.
    # We map posed voxel centers back to old frame and nearest-neighbor match to vanilla centers.
    # (In an ideal world, the transforms would map exactly to voxel centers, but resampling/voxelization differs.)
    centers_pose_to_old = apply_T(centers_pose_t, T_pose_to_old).detach().cpu().numpy()
    centers_old_np      = centers_old_t.detach().cpu().numpy()
    from scipy.spatial import cKDTree

    dx = 2.0 / args.res

    tree_old = cKDTree(centers_old_np)
    dists, nn = tree_old.query(centers_pose_to_old, k=1)

    matched_old = centers_old_np[nn]                  # [N_pos,3]
    diff_old = centers_pose_to_old - matched_old      # [N_pos,3]

    # robust estimate: median shift (in OLD frame)
    delta_old = np.median(diff_old, axis=0)

    print("[residual shift old] delta_old =", delta_old, " (voxels:", delta_old / dx, ")")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centers_old_np)
        dists, nn = tree.query(centers_pose_to_old, k=1)
        nn = nn.astype(np.int64)
        dists = dists.astype(np.float32)
    except Exception as e:
        raise RuntimeError("scipy not available for KDTree; install scipy or add your own NN matcher.") from e

    # compute cosine similarity between posed features and matched vanilla features
    feat_old_matched = feat_old[nn.to(torch.long)] if isinstance(nn, torch.Tensor) else feat_old[torch.from_numpy(nn).to(device)]
    # feat_old is [N_old,C], nn indexes vanilla for each posed point => [N_pos,C]
    sim = cosine_sim(feat_pos, feat_old_matched).detach().cpu().numpy()

    print(f"[NN match] mean NN distance (pose->old): {float(dists.mean()):.6f}  max: {float(dists.max()):.6f}")
    print(f"[feat cosine] mean: {float(sim.mean()):.4f}  median: {float(np.median(sim)):.4f}  p10: {float(np.percentile(sim,10)):.4f}")

    # quick histogram
    plt.figure(figsize=(5,3))
    plt.hist(sim, bins=50)
    plt.title("Cosine similarity: posed features vs NN-matched vanilla features")
    plt.xlabel("cosine sim")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig('fig3.png', dpi=160)
    # plt.show()

    # If the pipeline is correct, you should see:
    # - UV invariance error ~ tiny
    # - alpha score high for both
    # - cosine similarity concentrated near 1.0 (not perfect, but should be high)


if __name__ == "__main__":
    main()
