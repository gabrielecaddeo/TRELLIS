from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        pack = super().get_instance(root, instance, view)
       
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

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

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
    
class ImageConditionedMixinRotation:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        pack = super().get_instance(root, instance, view)
       
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

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

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
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
    
    def rgba_to_rgb_tensor(pose_img_rgba: Image.Image) -> torch.Tensor:
        arr = np.asarray(pose_img_rgba)            # HxWx4 (uint8)
        rgb = arr[..., :3].astype(np.float32) / 255.0
        a   = (arr[..., 3:4].astype(np.float32) / 255.0)
        rgb = rgb * a                              # premultiply over black
        return torch.from_numpy(rgb.transpose(2,0,1)).contiguous()  # [3,H,W]
    
    def get_instance2(self, root, instance_name):
        image_root = os.path.join(root, 'renders_cond', instance_name)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            meta_all = json.load(f)

        view = np.random.randint(len(meta_all['frames']))
        fr   = meta_all['frames'][view]
        image_path = os.path.join(image_root, fr['file_path'])
        image_rgba = Image.open(image_path).convert('RGBA')

        # Load pose2d_meta that you saved during data generation
        posed_name = f"{instance_name}_f{view:03d}"
        with open(os.path.join(root, 'POSED', instance_name, f"{posed_name}_meta.json")) as f:
            posed_meta = json.load(f)

        # We only need the 2D pose block:
        pose2d_meta = posed_meta.get("pose2d_meta", None)
        if pose2d_meta is None:
            # fallback or raise
            raise RuntimeError("pose2d_meta not found; regenerate posed metadata")

        # Rebuild pose-faithful conditioning image
        cond_img = self.collage_from_meta_bbox_preserve_aspect(
            image_rgba=image_rgba,
            pose2d_meta=pose2d_meta,
            out_size=self.image_size
        )
        cond_tensor = self.rgba_to_rgb_tensor(cond_img)

        pack = {}
        pack['cond'] = cond_tensor
        pack['frame_id'] = view
        pack['instance'] = instance_name
        return pack