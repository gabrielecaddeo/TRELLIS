import os
from PIL import Image
import json
import numpy as np
import pandas as pd
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase


class SparseFeat2Render(StandardDatasetBase):
    """
    SparseFeat2Render dataset.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'feature_{self.model}']]
        stats['With features'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    def _get_feat(self, root, instance):
        DATA_RESOLUTION = 64
        feats_path = os.path.join(root, 'features', self.model, f'{instance}.npz')
        feats = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats['indices']).int()
        feats = torch.tensor(feats['patchtokens']).float()
        
        if self.resolution != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // self.resolution
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean'
            )
        
        return {
            'coords': coords,
            'feats': feats,
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['feats'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        pack['image'] = torch.stack([b['image'] for b in batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])

        return pack

    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        return {
            **image,
            **feat,
        }


import glob
class SparseFeat2RenderPose(StandardDatasetBase):
    def __init__(
        self,
        roots: str,
        image_size: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,

        # NEW:
        use_pose_features: bool = True,          # if False, keep old behavior
        pose_sampling: str = "random",           # "random" or "first"
        vanilla_mix_prob: float = 0.0,           # e.g. 0.1 to sometimes use gold vanilla
        min_count: int = 0,                      # filter voxels with count < min_count (0 disables)
        cache_pose_lists: bool = True,           # build pose tag list once
    ):
        self.image_size = image_size
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        self.use_pose_features = use_pose_features
        self.pose_sampling = pose_sampling
        self.vanilla_mix_prob = float(vanilla_mix_prob)
        self.min_count = int(min_count)
        self.cache_pose_lists = bool(cache_pose_lists)

        # internal cache: { (root, instance) : [pose_tags...] }
        self._pose_tags_cache = {}

        super().__init__(roots)

    def _list_pose_tags(self, root, instance):
        pose_dir = os.path.join(root, "data_pose_norm", instance)
        feat_dir = os.path.join(root, "features", self.model)
        meta_paths = sorted(glob.glob(os.path.join(pose_dir, f"{instance}_f*_meta.json")))
        tags = [os.path.basename(p).replace("_meta.json", "") for p in meta_paths]
        tags = [t for t in tags if os.path.exists(os.path.join(feat_dir, f"{t}.npz"))]
        return tags

    def _pick_pose_tag(self, tags):
        if len(tags) == 0:
            return None
        if self.pose_sampling == "first":
            return tags[0]
        return tags[np.random.randint(len(tags))]

    def _load_pose_pack(self, root, instance, pose_tag):
        feat_path = os.path.join(root, "features", self.model, f"{pose_tag}.npz")
        meta_path = os.path.join(root, "data_pose_norm", instance, f"{pose_tag}_meta.json")
        pack = np.load(feat_path, allow_pickle=True)
        meta = json.load(open(meta_path, "r"))

        coords = torch.tensor(pack["indices"]).int()          # posed indices (N,3)
        feats  = torch.tensor(pack["patchtokens"]).float()    # (N,C)
        count  = torch.tensor(pack["count"]).int() if "count" in pack.files else None

        # pose params needed for posed->world in decoder/trainer
        pose = meta["pose"]
        can  = meta["canonical"]
        pose_params = {
            "R_row": torch.tensor(pose["R_fixed"]).float(),      # (3,3)
            "s_aug": torch.tensor(pose["s_aug"]).float(),        # ()
            "t_aug": torch.tensor(pose["t_aug"]).float(),        # (3,)
            "c0":    torch.tensor(pose["c0"]).float(),           # (3,)
            "s_norm": torch.tensor(can["s_norm"]).float(),       # ()
            "c_norm": torch.tensor(can["c_norm"]).float(),       # (3,)
        }
        return coords, feats, count, pose_params

    def _load_vanilla_pack(self, root, instance):
        feat_path = os.path.join(root, "features", self.model, f"{instance}.npz")
        pack = np.load(feat_path, allow_pickle=True)
        coords = torch.tensor(pack["indices"]).int()
        feats  = torch.tensor(pack["patchtokens"]).float()
        count  = torch.tensor(pack["count"]).int() if "count" in pack.files else None
        return coords, feats, count

    def _get_feat(self, root, instance):
        DATA_RES = 64
        assert self.resolution <= DATA_RES and (DATA_RES % self.resolution == 0)

        use_vanilla = (not self.use_pose) or (self.vanilla_mix_prob > 0 and np.random.rand() < self.vanilla_mix_prob)

        pose_params = None
        if use_vanilla:
            coords, feats, count = self._load_vanilla_pack(root, instance)
        else:
            tags = self._list_pose_tags(root, instance)
            pose_tag = self._pick_pose_tag(tags)
            if pose_tag is None:
                coords, feats, count = self._load_vanilla_pack(root, instance)
            else:
                coords, feats, count, pose_params = self._load_pose_pack(root, instance, pose_tag)

        # optional count filtering
        if count is not None and self.min_count > 0:
            m = count >= self.min_count
            coords, feats = coords[m], feats[m]

        # downsample if needed (same as your old code)
        if self.resolution != DATA_RES:
            factor = DATA_RES // self.resolution
            coords = coords // factor
            coords_u, inv = coords.unique(return_inverse=True, dim=0)
            feats_u = torch.scatter_reduce(
                torch.zeros(coords_u.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=inv.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean',
                include_self=False,
            )
            coords, feats = coords_u, feats_u

        out = {"coords": coords, "feats": feats}
        if pose_params is not None:
            out["pose_params"] = pose_params  # dict of tensors
        else:
            out["pose_params"] = None
        return out

    @staticmethod
    def collate_fn(batch):
        pack = {}

        # sparse feats
        coords = []
        feats = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
            feats.append(b['feats'])
        pack['feats'] = SparseTensor(coords=torch.cat(coords, dim=0), feats=torch.cat(feats, dim=0))

        # images/cams
        pack['image'] = torch.stack([b['image'] for b in batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])

        # pose params: stack per-sample (and mark which samples are posed)
        posed_mask = torch.tensor([b["pose_params"] is not None for b in batch], dtype=torch.bool)
        pack["posed_mask"] = posed_mask

        def stack_param(name, default):
            vals = []
            for b in batch:
                if b["pose_params"] is None:
                    vals.append(default.clone())
                else:
                    vals.append(b["pose_params"][name])
            return torch.stack(vals, dim=0)

        # defaults = identity transform for vanilla samples
        I3 = torch.eye(3)
        z3 = torch.zeros(3)
        one = torch.tensor(1.0)
        pack["R_row"]  = stack_param("R_row", I3)     # (B,3,3)
        pack["s_aug"]  = stack_param("s_aug", one)    # (B,)
        pack["t_aug"]  = stack_param("t_aug", z3)     # (B,3)
        pack["c0"]     = stack_param("c0",  z3)       # (B,3)
        pack["s_norm"] = stack_param("s_norm", one)   # (B,)
        pack["c_norm"] = stack_param("c_norm", z3)    # (B,3)

        return pack

    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        feat = self._get_feat(root, instance)
        return {**image, **feat}