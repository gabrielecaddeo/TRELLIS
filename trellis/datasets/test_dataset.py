import torch 
from PIL import Image
import numpy as np
import json
import os


class TestDatasetConditioned:
    def __init__(self, image_size=518, latent_model: str = 'vae_ABO_HSSD_3D_rot_outer_rim_0140000'):
        self.image_size = image_size
        self.latent_model = latent_model
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance_name, view = None):
        image_root = os.path.join(root, 'renders_cond', instance_name)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            meta_all = json.load(f)

        if view is None:
            view = np.random.randint(len(meta_all['frames']))
        
        fr   = meta_all['frames'][view]
        image_path = os.path.join(image_root, fr['file_path'])
        image_rgba = Image.open(image_path).convert('RGBA')

        mask_hand_path = os.path.join(image_root, f"{view:03d}_mask_1.png")
        mask_obj_path  = os.path.join(image_root, f"{view:03d}_mask_2.png")

        pack = self.get_tensors(root, instance_name, view)

        posed_name = f"{instance_name}_f{view:03d}"
        with open(os.path.join(root, 'data_pose_norm', instance_name, f"{posed_name}_meta.json")) as f:
            posed_meta = json.load(f)
        pose2d_meta = posed_meta.get("pose2d_meta", None)
        if pose2d_meta is None:
            raise RuntimeError("pose2d_meta not found; regenerate posed metadata")

        # --- compute transform ONCE from RGBA ---
        tf = self._compute_sprite_transform(image_rgba, pose2d_meta, self.image_size)

        # --- apply to RGBA cond image ---
        cond_img = self.collage_from_meta_bbox_preserve_aspect(
            image_rgba=image_rgba,
            pose2d_meta=pose2d_meta,
            out_size=self.image_size
        )
        cond_tensor = self.rgba_to_rgb_tensor(cond_img)
        pack['cond'] = cond_tensor.unsqueeze(0)  # [1,3,H,W]

        # --- apply SAME transform to masks ---
        mask_hand_img = Image.open(mask_hand_path).convert('L')
        mask_obj_img  = Image.open(mask_obj_path).convert('L')

        mask_hand_canvas = self.apply_transform_to_mask(mask_hand_img, tf, self.image_size)
        mask_obj_canvas  = self.apply_transform_to_mask(mask_obj_img,  tf, self.image_size)

        # turn masks into tensors [1,H,W]
        hand_arr = np.asarray(mask_hand_canvas).astype(np.float32) / 255.0
        obj_arr  = np.asarray(mask_obj_canvas).astype(np.float32) / 255.0

        pack['mask_hand'] = torch.from_numpy(hand_arr).unsqueeze(0)
        pack['mask_obj']  = torch.from_numpy(obj_arr).unsqueeze(0)

        cond_masked = pack['cond'] * pack['mask_obj']       # broadcast over channel dim → [3,H,W]
        pack['cond_mask'] = cond_masked        # [1,3,H,W]
        pack['frame_id'] = view
        pack['instance'] = instance_name

        contacts_indices = np.load(os.path.join(root, 'data_pose_norm', instance_name, 'contacts', instance_name + f'_f{view:03d}_contact_coords.npy'))
        contact_grid = np.zeros((64,64,64), dtype=np.float32)
        contact_grid[contacts_indices[:,0], contacts_indices[:,1], contacts_indices[:,2]] = 1.0
        contact_sdf = np.load(os.path.join(root, 'data_pose_norm', instance_name, 'contacts', instance_name + f'_f{view:03d}_dist_to_contact.npy'))
        pack['touch'] = torch.cat([torch.from_numpy(contact_grid).unsqueeze(0), torch.from_numpy(contact_sdf).unsqueeze(0)], dim=0).unsqueeze(0)  #
        return pack

    def get_tensors(self, root, instance, n_view):
        latent = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_{n_view}__object.npz'))
        z = torch.tensor(latent['mean']).float()
        latent_hand = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_{n_view}__hand.npz'))
        z_hand = torch.tensor(latent_hand['mean']).float()
        pack = {
            'x_0': z.unsqueeze(0),
            'x0_hand': z_hand.unsqueeze(0),
        }
        return pack
    def rgba_to_rgb_tensor(self, pose_img_rgba: Image.Image) -> torch.Tensor:
        arr = np.asarray(pose_img_rgba)            # HxWx4 (uint8)
        rgb = arr[..., :3].astype(np.float32) / 255.0
        a   = (arr[..., 3:4].astype(np.float32) / 255.0)
        rgb = rgb * a                              # premultiply over black
        return torch.from_numpy(rgb.transpose(2,0,1)).contiguous()  # [3,H,W]
    
    def _compute_sprite_transform(
        self,
        image_rgba: Image.Image,
        pose2d_meta: dict,
        out_size: int,
    ):
        W = H = int(out_size)
        res      = int(pose2d_meta["res"])
        bbox_xy  = pose2d_meta["bbox_xy"]

        if bbox_xy is None:
            return None  # empty case

        xmin, ymin, xmax, ymax = bbox_xy

        # voxel bbox -> pixel bbox
        px_per_vox = float(W) / float(res)
        xmin_px = int(round(xmin * px_per_vox))
        ymin_px = int(round(ymin * px_per_vox))
        w_px_box = max(1, int(round((xmax - xmin + 1) * px_per_vox)))
        h_px_box = max(1, int(round((ymax - ymin + 1) * px_per_vox)))

        # crop sprite region from RGBA alpha
        A  = np.asarray(image_rgba.getchannel("A"))
        ys, xs = np.nonzero(A > 0)
        if len(xs) == 0:
            return None

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        sw = x1 - x0 + 1
        sh = y1 - y0 + 1

        # uniform scale
        scale = max(1e-6, min(w_px_box / sw, h_px_box / sh))
        new_w = max(1, int(round(sw * scale)))
        new_h = max(1, int(round(sh * scale)))

        # center within the target box
        dx_px = (w_px_box - new_w) // 2
        dy_px = (h_px_box - new_h) // 2
        paste_x = xmin_px + dx_px
        paste_y = ymin_px + dy_px

        paste_x = max(-new_w, min(W, paste_x))
        paste_y = max(-new_h, min(H, paste_y))

        return {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "new_w": new_w, "new_h": new_h,
            "paste_x": paste_x, "paste_y": paste_y,
            "W": W, "H": H,
        }

    def collage_from_meta_bbox_preserve_aspect(
        self,
        image_rgba: Image.Image,
        pose2d_meta: dict,
        out_size: int = 1024,
    ):
        assert image_rgba.mode == "RGBA"
        tf = self._compute_sprite_transform(image_rgba, pose2d_meta, out_size)
        W = H = int(out_size)
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        if tf is None:
            return canvas

        sprite = image_rgba.crop((tf["x0"], tf["y0"], tf["x1"] + 1, tf["y1"] + 1))
        sprite_resized = sprite.resize((tf["new_w"], tf["new_h"]), Image.Resampling.LANCZOS)

        canvas.alpha_composite(sprite_resized, dest=(tf["paste_x"], tf["paste_y"]))
        return canvas

    def apply_transform_to_mask(
        self,
        mask_img: Image.Image,   # e.g. "L"
        tf: dict,
        out_size: int,
    ):
        W = H = int(out_size)
        # background 0 for masks
        canvas = Image.new(mask_img.mode, (W, H), 0)

        if tf is None:
            return canvas

        sprite = mask_img.crop((tf["x0"], tf["y0"], tf["x1"] + 1, tf["y1"] + 1))
        sprite_resized = sprite.resize(
            (tf["new_w"], tf["new_h"]),
            Image.Resampling.NEAREST
        )
        canvas.paste(sprite_resized, (tf["paste_x"], tf["paste_y"]))
        return canvas
