import torch 
from PIL import Image, ImageFilter
import numpy as np
import json
import os


class TestDatasetConditioned:
    def __init__(self, image_size=518, latent_model: str = 'vae_final_all_resume_2_0300000'): #vae_ABO_HSSD_3D_rot_outer_rim_0140000/
        self.image_size = image_size
        self.latent_model = latent_model
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats


    def _shift_mask(self, mask_img: Image.Image, dx: int, dy: int) -> Image.Image:
        """Shift mask with zero padding (no wrap-around). mask_img is mode 'L'."""
        arr = np.asarray(mask_img)
        H, W = arr.shape
        out = np.zeros_like(arr)
        
        x_src0 = max(0, -dx)
        x_src1 = min(W, W - dx)
        y_src0 = max(0, -dy)
        y_src1 = min(H, H - dy)
        
        x_dst0 = max(0, dx)
        x_dst1 = min(W, W + dx)
        y_dst0 = max(0, dy)
        y_dst1 = min(H, H + dy)
        
        out[y_dst0:y_dst1, x_dst0:x_dst1] = arr[y_src0:y_src1, x_src0:x_src1]
        return Image.fromarray(out, mode=mask_img.mode)
    
    def jitter_mask_shift(self, mask_img: Image.Image, p: float, max_shift: int, rng) -> Image.Image:
        """
        With prob p, shift entire mask by random dx,dy in [-max_shift, +max_shift]
        (diagonals allowed), excluding (0,0).
        """
        if max_shift <= 0 or rng.random() >= p:
            return mask_img

        # sample non-zero shift
        while True:
            dx = int(rng.integers(-max_shift, max_shift + 1))
            dy = int(rng.integers(-max_shift, max_shift + 1))
            if dx != 0 or dy != 0:
                break

        return self._shift_mask(mask_img, dx, dy)

    def boundary_corrupt(self,
                         mask_img: Image.Image,
                         p_apply: float,
                         p_pixel: float,
                         radius_range=(1, 1),
                         mode: str = "erode",
                         rng=None,
                         thresh: int = 127) -> Image.Image:
        """
        Corrupt ONLY the boundary ring of a (mostly) binary mask.
        - p_apply: probability to apply boundary corruption at all
        - p_pixel: probability to corrupt each boundary pixel
        - radius_range: boundary thickness control via min/max filter size
        - mode:
        'erode'  -> boundary pixels turned OFF (shrinks locally)
        'dilate' -> boundary pixels turned ON  (grows locally)
        'flip'   -> boundary pixels inverted
        'random' -> boundary pixels set randomly to 0/1
        """
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() >= p_apply:
            return mask_img

        r0, r1 = radius_range
        r0 = max(1, int(r0))
        r1 = max(r0, int(r1))
        r = int(rng.integers(r0, r1 + 1))
        k = 2 * r + 1  # odd kernel size for PIL Min/MaxFilter
        
        # binarize
        arr = np.asarray(mask_img)
        m = (arr > thresh)
        
        # approximate erosion/dilation via rank filters
        eroded  = (np.asarray(mask_img.filter(ImageFilter.MinFilter(size=k))) > thresh)
        dilated = (np.asarray(mask_img.filter(ImageFilter.MaxFilter(size=k))) > thresh)
        
        # boundary ring: pixels that change between eroded and dilated
        boundary = (dilated ^ eroded)
        
        # choose which boundary pixels to corrupt
        corrupt = boundary & (rng.random(boundary.shape) < p_pixel)
        
        out = m.copy()
        if mode == "erode":
            out[corrupt] = False
        elif mode == "dilate":
            out[corrupt] = True
        elif mode == "flip":
            out[corrupt] = ~out[corrupt]
        elif mode == "random":
            out[corrupt] = (rng.random(np.count_nonzero(corrupt)) < 0.5)
        else:
            raise ValueError("mode must be one of: 'erode','dilate','flip','random'")

        return Image.fromarray(out.astype(np.uint8) * 255, mode="L")

    def ablate_object_mask(self,
                           mask_obj_canvas: Image.Image,
                           preset: str = "small",
                           seed: int = None,
                           coupled: bool = True,
                           do_shift: bool = True,
                           do_boundary: bool = True,
                           boundary_mode: str = "erode") -> Image.Image:
        """
        Two noise regimes:
        - small: small shift + thin boundary corruption
        - big:   bigger shift + thicker/more aggressive boundary corruption
        
        coupled=True  -> one Bernoulli decides applying both (simultaneously)
        coupled=False -> shift and boundary decisions are independent
        """
        rng = np.random.default_rng(seed)
        
        if preset == "small":
            cfg = dict(
                p_joint=0.35,
                p_shift=0.35, max_shift=1,
                p_boundary=0.35, p_pixel=0.25, radius_range=(1, 1),
            )
        elif preset == "big":
            cfg = dict(
                p_joint=0.70,
                p_shift=0.70, max_shift=3,           # try 2..5 depending on your image_size
                p_boundary=0.70, p_pixel=0.50, radius_range=(2, 3),
            )
        else:
            raise ValueError("preset must be 'small' or 'big'")

        out = mask_obj_canvas
        
        if coupled:
            apply_all = (rng.random() < cfg["p_joint"])
            if apply_all and do_shift:
                out = self.jitter_mask_shift(out, p=1.0, max_shift=cfg["max_shift"], rng=rng)
            if apply_all and do_boundary:
                out = self.boundary_corrupt(out, p_apply=1.0, p_pixel=cfg["p_pixel"],
                                        radius_range=cfg["radius_range"], mode=boundary_mode, rng=rng)
        else:
            if do_shift:
                out = self.jitter_mask_shift(out, p=cfg["p_shift"], max_shift=cfg["max_shift"], rng=rng)
            if do_boundary:
                out = self.boundary_corrupt(out, p_apply=cfg["p_boundary"], p_pixel=cfg["p_pixel"],
                                        radius_range=cfg["radius_range"], mode=boundary_mode, rng=rng)

        return out


    def jitter_contacts_one_voxel(self, contacts_indices, grid_shape=(64, 64, 64), z=0.1, seed=None,
                                  boundary_mode="keep"):
        """
        Move each contact voxel by at most 1 voxel in a random (possibly diagonal) direction
        with probability z.
        
        boundary_mode:
        - "keep": if the move goes out of bounds, keep the original voxel
        - "clip": clip coordinates to stay in bounds (can bias toward borders)
        """
        contacts_indices = np.asarray(contacts_indices, dtype=np.int64)
        rng = np.random.default_rng(seed)
        
        # Optional: remove duplicates up front
        if len(contacts_indices) == 0:
            return contacts_indices
        
        # Decide which ones move
        move_mask = rng.random(len(contacts_indices)) < z
        moved = contacts_indices.copy()
        
        # 26 possible offsets (including diagonals), excluding (0,0,0)
        offsets = np.array([(dx, dy, dz)
                            for dx in (-1, 0, 1)
                            for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)
                            if not (dx == 0 and dy == 0 and dz == 0)], dtype=np.int64)

        n_move = int(move_mask.sum())
        if n_move > 0:
            sampled_offsets = offsets[rng.integers(0, len(offsets), size=n_move)]
            proposed = moved[move_mask] + sampled_offsets

            if boundary_mode == "keep":
                in_bounds = (
                    (proposed[:, 0] >= 0) & (proposed[:, 0] < grid_shape[0]) &
                    (proposed[:, 1] >= 0) & (proposed[:, 1] < grid_shape[1]) &
                    (proposed[:, 2] >= 0) & (proposed[:, 2] < grid_shape[2])
                )
                # apply only valid moves; invalid ones remain original
                moved_subset = moved[move_mask]
                moved_subset[in_bounds] = proposed[in_bounds]
                moved[move_mask] = moved_subset
                
            elif boundary_mode == "clip":
                proposed[:, 0] = np.clip(proposed[:, 0], 0, grid_shape[0] - 1)
                proposed[:, 1] = np.clip(proposed[:, 1], 0, grid_shape[1] - 1)
                proposed[:, 2] = np.clip(proposed[:, 2], 0, grid_shape[2] - 1)
                moved[move_mask] = proposed
            else:
                raise ValueError("boundary_mode must be 'keep' or 'clip'")

        # Remove duplicates after moving (collisions are fine for a binary grid)
        moved = np.unique(moved, axis=0)
        return moved

    
    def get_instance(self, root, instance_name, view = None):
        image_root = os.path.join(root, 'renders_cond', instance_name)
        #with open(os.path.join(image_root, 'transforms.json')) as f:
        #    meta_all = json.load(f)

        if view is None:
            view = np.random.randint(len(meta_all['frames']))
        
        #fr   = meta_all['frames'][view]
        image_path = os.path.join(image_root, '000.png')
        image_rgba = Image.open(image_path).convert('RGBA')

        mask_hand_path = os.path.join(image_root, f"{view:03d}_mask1.png")
        mask_obj_path  = os.path.join(image_root, f"{view:03d}_mask2.png")

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
        # mask_hand_img = Image.open(mask_hand_path).convert('L')
        # mask_obj_img  = Image.open(mask_obj_path).convert('L')

        # mask_hand_canvas = self.apply_transform_to_mask(mask_hand_img, tf, self.image_size)
        # mask_obj_canvas  = self.apply_transform_to_mask(mask_obj_img,  tf, self.image_size)
        # mask_obj_canvas = self.ablate_object_mask(
        #     mask_obj_canvas,
        #     preset="small",
        #     seed=123 + view,        # vary per view; you can also mix instance hash
        #     coupled=True,           # True = simultaneous / False = independent
        #     do_shift=True,
        #     do_boundary=True,
        #     boundary_mode="erode",  # or "flip"/"dilate"/"random"
        # )
        ### START
        mask_hand_img = Image.open(mask_hand_path).convert('L')
        mask_obj_img  = Image.open(mask_obj_path).convert('L')
        
        mask_hand_canvas = self.apply_transform_to_mask(mask_hand_img, tf, self.image_size)
        mask_obj_canvas_clean = self.apply_transform_to_mask(mask_obj_img, tf, self.image_size)
        
        # ---- APPLY ABLATION (jitter + boundary corruption) ----
        mask_obj_canvas_noisy = self.ablate_object_mask(
            mask_obj_canvas_clean,
            preset="big",        # or "big"
            seed=123 + view,
            coupled=True,          # True=simultaneous, False=independent
            do_shift=True,
            do_boundary=True,
            boundary_mode="erode", # or "flip"/"dilate"/"random"
        )

        # ---- DEBUG SAVE (before/after) ----
        # debug_dir = os.path.join(root, "debug_mask_ablation", instance_name)
        # os.makedirs(debug_dir, exist_ok=True)
        
        # mask_obj_canvas_clean.save(os.path.join(debug_dir, f"{view:03d}_mask_obj_clean.png"))
        # mask_obj_canvas_noisy.save(os.path.join(debug_dir, f"{view:03d}_mask_obj_noisy.png"))
        
        # # Optionally save masked RGB images too (very useful to see the effect)
        # cond_rgb = cond_img.convert("RGB")  # cond_img is your transformed RGBA canvas
        # cond_arr = np.asarray(cond_rgb).astype(np.float32)
        
        # m_clean = (np.asarray(mask_obj_canvas_clean).astype(np.float32) / 255.0)
        # m_noisy = (np.asarray(mask_obj_canvas_noisy).astype(np.float32) / 255.0)
        
        # masked_clean = (cond_arr * m_clean[..., None]).clip(0, 255).astype(np.uint8)
        # masked_noisy = (cond_arr * m_noisy[..., None]).clip(0, 255).astype(np.uint8)
        
        # Image.fromarray(masked_clean).save(os.path.join(debug_dir, f"{view:03d}_cond_masked_clean.png"))
        # Image.fromarray(masked_noisy).save(os.path.join(debug_dir, f"{view:03d}_cond_masked_noisy.png"))
        
        # IMPORTANT: use the noisy one downstream if you want the ablation to take effect
        #mask_obj_canvas = mask_obj_canvas_noisy
        mask_obj_canvas = mask_obj_canvas_clean

        ### DONE
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
        # contacts_indices_noisy = self.jitter_contacts_one_voxel(
        #     contacts_indices,
        #     grid_shape=(64, 64, 64),
        #     z=0.7,
        #     seed=123,
        #     boundary_mode="keep",
        # )
        
        contact_grid = np.zeros((64,64,64), dtype=np.float32)
        contact_grid[contacts_indices[:,0], contacts_indices[:,1], contacts_indices[:,2]] = 1.0
        contact_sdf = np.load(os.path.join(root, 'data_pose_norm', instance_name, 'contacts', instance_name + f'_f{view:03d}_dist_to_contact.npy'))
        pack['touch'] = torch.cat([torch.from_numpy(contact_grid).unsqueeze(0), torch.from_numpy(contact_sdf).unsqueeze(0)], dim=0).unsqueeze(0)  #
        return pack

    def get_tensors(self, root, instance, n_view):
        latent = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_f{n_view:02d}__object.npz'))
        z = torch.tensor(latent['mean']).float()
        latent_hand = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_f{n_view:02d}__hand.npz'))
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
