# extract_features_pose_norm_fast.py
import os, json, glob, argparse, copy
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import torch.nn.functional as F

import utils3d  # your project

torch.set_grad_enabled(False)


# -----------------------------
# Image loading (same normalization)
# -----------------------------
def load_image_518(path):
    img = Image.open(path).convert("RGBA").resize((518, 518), Image.Resampling.LANCZOS)
    arr = (np.asarray(img).astype(np.float32) / 255.0)  # H,W,4
    rgb = arr[..., :3]
    a   = arr[..., 3:4]
    rgb = rgb * a  # premultiply alpha
    x = torch.from_numpy(rgb).permute(2, 0, 1).float()  # 3,518,518

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x


def view_mats(frame, device="cpu"):
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
    c2w[:3, 1:3] *= -1
    extr = torch.inverse(c2w)  # world->cam
    fov = torch.tensor(frame["camera_angle_x"], dtype=torch.float32, device=device)
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov).to(device)
    return extr, intr


# -----------------------------
# DINOv2 patchtokens
# -----------------------------
def load_dinov2(model_name, device):
    m = torch.hub.load("facebookresearch/dinov2", model_name)
    m.eval().to(device)
    return m


@torch.no_grad()
def dinov2_patchtokens_map(model, images_B3HW):
    feats = model(images_B3HW, is_training=True)
    tokens = feats["x_prenorm"][:, model.num_register_tokens + 1:]  # [B, Hp*Wp, C]
    B, T, C = tokens.shape
    Hp = Wp = 518 // 14
    patch = tokens.permute(0, 2, 1).reshape(B, C, Hp, Wp)  # [B,C,Hp,Wp]
    return patch


@torch.no_grad()
def sample_patchtokens(patchtokens_map_BCHW, uv01_BN2):
    # Make grid dtype match input dtype (IMPORTANT for CUDA grid_sample)
    uv01_BN2 = uv01_BN2.to(dtype=patchtokens_map_BCHW.dtype)

    grid = uv01_BN2.unsqueeze(2) * 2.0 - 1.0   # [B,N,1,2]
    out = F.grid_sample(
        patchtokens_map_BCHW, grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    )  # [B,C,N,1]
    out = out.squeeze(-1).permute(0, 2, 1)     # [B,N,C]
    return out



# -----------------------------
# Pose / canonical transforms
# -----------------------------
def idxs_to_centers_unit2(idxs, origin, voxel_size):
    idxs = idxs.astype(np.float32)
    origin = np.array(origin, dtype=np.float32)
    centers = origin + (idxs + 0.5) * float(voxel_size)
    return centers  # (N,3)


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


# -----------------------------
# Projection helper
# -----------------------------
def project_uv01(points_world_N3, extr_B44, intr_B33):
    uv = utils3d.torch.project_cv(points_world_N3, extr_B44, intr_B33)[0]  # [B,N,2] in [0,1]
    return uv


# -----------------------------
# Loader (parallel like vanilla)
# -----------------------------
def load_views_data(root, sha, max_workers=16):
    tf_path = os.path.join(root, "renders", sha, "transforms.json")
    meta = json.load(open(tf_path, "r"))
    frames = meta["frames"]

    render_dir = os.path.join(root, "renders", sha)

    def worker(frame):
        img_path = os.path.join(render_dir, frame["file_path"])
        if not os.path.exists(img_path):
            return None
        try:
            img = load_image_518(img_path)  # CPU tensor 3x518x518
        except Exception:
            return None
        extr, intr = view_mats(frame, device="cpu")  # CPU tensors
        return {"image": img, "extrinsics": extr, "intrinsics": intr}

    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for d in ex.map(worker, frames):
            if d is not None:
                data.append(d)

    return data  # list of dicts


def list_objects_pose_norm(root):
    pose_root = os.path.join(root, "data_pose_norm")
    if not os.path.isdir(pose_root):
        return []
    return sorted([d for d in os.listdir(pose_root) if os.path.isdir(os.path.join(pose_root, d))])


# -----------------------------
# Main per-object compute (FAST)
# -----------------------------
@torch.no_grad()
def process_one_object_fast(root, sha, model, feature_name, batch_size=16, res=64, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    render_dir = os.path.join(root, "renders", sha)
    pose_dir   = os.path.join(root, "data_pose_norm", sha)
    if not (os.path.isdir(render_dir) and os.path.isdir(pose_dir)):
        return

    # pose meta paths
    meta_paths = sorted(glob.glob(os.path.join(pose_dir, f"{sha}_f*_meta.json")))
    if len(meta_paths) == 0:
        return

    # output folder
    feat_root = os.path.join(root, "features", feature_name)
    os.makedirs(feat_root, exist_ok=True)

    # ---- 1) Load all views ONCE (threaded) ----
    views = load_views_data(root, sha, max_workers=64)
    if len(views) == 0:
        return

    # ---- 2) Run DINO ONCE per object, cache patch_maps per view-batch on CPU ----
    # We cache (patch_map, extr, intr) batchwise so we can reuse for all 24 poses
    cached = []
    n_patch = 518 // 14

    # (optional speed knobs)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for i in range(0, len(views), batch_size):
        batch = views[i:i+batch_size]
        B = len(batch)

        imgs = torch.stack([d["image"] for d in batch], dim=0).to(device, non_blocking=True)  # [B,3,518,518]
        extr = torch.stack([d["extrinsics"] for d in batch], dim=0).to(device, non_blocking=True)  # [B,4,4]
        intr = torch.stack([d["intrinsics"] for d in batch], dim=0).to(device, non_blocking=True)  # [B,3,3]

        # AMP helps a lot on modern GPUs
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                patch_map = dinov2_patchtokens_map(model, imgs)  # [B,C,Hp,Wp]
        else:
            patch_map = dinov2_patchtokens_map(model, imgs)

        # move cache to CPU float16 to keep VRAM low
        cached.append({
            "patch_map": patch_map.detach().to("cpu", dtype=torch.float16),
            "extr": extr.detach().to("cpu"),
            "intr": intr.detach().to("cpu"),
        })

        # free VRAM for next batch
        del imgs, extr, intr, patch_map
        if device.type == "cuda":
            torch.cuda.synchronize()

    # free views (images) now — we don’t need them anymore
    del views

    # ---- 3) For each pose: project + sample using cached patch maps (NO DINO rerun) ----
    for meta_path in meta_paths:
        pose_tag = os.path.basename(meta_path).replace("_meta.json", "")  # sha_f000
        idx_path = os.path.join(pose_dir, "idxs", f"{pose_tag}.npy")
        if not os.path.exists(idx_path):
            continue

        out_npz = os.path.join(feat_root, f"{pose_tag}.npz")
        if os.path.exists(out_npz):
            continue

        posed_meta = json.load(open(meta_path, "r"))

        # canonical norm (saved by you)
        s_norm = float(posed_meta["canonical"]["s_norm"])
        c_norm = np.array(posed_meta["canonical"]["c_norm"], dtype=np.float32)

        # pose params
        pose = posed_meta["pose"]
        R_row = np.array(pose["R_fixed"], dtype=np.float32)
        s_aug = float(pose["s_aug"])
        t_aug = np.array(pose["t_aug"], dtype=np.float32)
        c0    = np.array(pose["c0"], dtype=np.float32)

        # grid params
        grid = posed_meta["grid"]
        origin = tuple(grid["origin"])
        voxel_size = float(grid["voxel_size"])  # ~ 2/res

        # points (posed unit2 -> canonical -> world)
        idxs = np.load(idx_path).astype(np.int32)  # (N,3)
        centers_pose = idxs_to_centers_unit2(idxs, origin=origin, voxel_size=voxel_size).astype(np.float32)
        centers_can  = posed_to_canonical_row(centers_pose, R_row, s_aug, t_aug, c0).astype(np.float32)
        centers_w    = canonical_to_render_world(centers_can, s_norm, c_norm).astype(np.float32)

        points_w = torch.from_numpy(centers_w).to(device)  # [N,3]
        N = points_w.shape[0]

        feat_sum = None
        count    = torch.zeros((N, 1), dtype=torch.float32, device=device)

        ones = torch.ones((N, 1), dtype=torch.float32, device=device)
        Pw_h = torch.cat([points_w, ones], dim=1)  # [N,4]

        for pack in cached:
            patch_map = pack["patch_map"].to(device, non_blocking=True)  # [B,C,Hp,Wp] float16
            extr = pack["extr"].to(device, non_blocking=True)            # [B,4,4]
            intr = pack["intr"].to(device, non_blocking=True)            # [B,3,3]
            B = patch_map.shape[0]

            # project
            uv01 = project_uv01(points_w, extr, intr)  # [B,N,2]

            # behind-camera rejection (z <= 0)
            Pc = (extr @ Pw_h.t()).transpose(1, 2)[..., :3]  # [B,N,3]
            z = Pc[..., 2:3]

            valid = (z > 1e-6) & \
                    (uv01[..., 0:1] >= 0.0) & (uv01[..., 0:1] <= 1.0) & \
                    (uv01[..., 1:2] >= 0.0) & (uv01[..., 1:2] <= 1.0)
            valid_f = valid.float()

            # valid_f should match feats dtype for multiplication
            valid_f = valid_f.to(dtype=patch_map.dtype)

            feats_bnC = sample_patchtokens(patch_map, uv01)  # half
            feats_bnC = feats_bnC * valid_f                  # half

            if feat_sum is None:
                C = feats_bnC.shape[-1]
                feat_sum = torch.zeros((N, C), dtype=torch.float32, device=device)

            # accumulate in float32
            feat_sum += feats_bnC.sum(dim=0).float()
            count    += valid_f.sum(dim=0).float()

        feat_mean = feat_sum / torch.clamp(count, min=1.0)  # [N,C]

        # save
        pack = {
            "indices": idxs.astype(np.uint8),
            "patchtokens": feat_mean.detach().cpu().numpy().astype(np.float16),
            "count": count.detach().cpu().numpy().astype(np.uint16).squeeze(1),
        }
        np.savez_compressed(out_npz, **pack)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dataset root containing renders/ and data_pose_norm/. Features will be written in output_dir/features/<model>/")
    parser.add_argument("--model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loader_workers", type=int, default=128,
                        help="How many objects to preload in parallel (RAM bound).")
    args = parser.parse_args()

    root = args.output_dir
    feature_name = args.model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(os.path.join(root, "features", feature_name), exist_ok=True)

    model = load_dinov2(args.model, device=device)

    shas = list_objects_pose_norm(root)
    print(f"Found {len(shas)} objects in data_pose_norm/")

    # Vanilla-like pipeline: object loader queue + main compute + saver thread
    load_queue = Queue(maxsize=128)

    def loader(sha):
        # Just enqueue sha; process_one_object_fast loads views internally with threads.
        load_queue.put(sha)

    with ThreadPoolExecutor(max_workers=args.loader_workers) as loader_ex:
        loader_ex.map(loader, shas)

        for _ in tqdm(range(len(shas)), desc="Objects"):
            sha = load_queue.get()
            try:
                process_one_object_fast(
                    root=root,
                    sha=sha,
                    model=model,
                    feature_name=feature_name,
                    batch_size=args.batch_size,
                    res=args.res,
                    device=str(device),
                )
            except Exception as e:
                print(f"[ERROR] {sha}: {e}")


if __name__ == "__main__":
    main()
