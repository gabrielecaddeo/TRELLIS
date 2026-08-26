import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin, ImageConditionedMixinRotation, ImageConditionedMixinRotationConditioned
from .. import models


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)


class SparseStructureLatentVisMixinSDF:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            mask = x_0[i, 0] <= 0
            coords = torch.nonzero(mask, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images), x_0


class SparseStructureLatentVisMixinSDFConditioned:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            mask = x_0[i, 0] <= 0
            coords = torch.nonzero(mask, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images), x_0

class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack
    

class SparseStructureLatentSDF(SparseStructureLatentVisMixinSDF, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance, n_view):
        latent = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_{n_view}.npz'))
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack


class SparseStructureLatentSDFConditioned(SparseStructureLatentVisMixinSDFConditioned, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance, n_view):
        latent = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_{n_view}__object.npz'))
        z = torch.tensor(latent['mean']).float()
        latent_hand = np.load(os.path.join(root, 'ss_latents_sdf_pose', self.latent_model, f'{instance}_{n_view}__hand.npz'))
        z_hand = torch.tensor(latent_hand['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
            z_hand = (z_hand - self.mean) / self.std

        pack = {
            'x_0': z,
            'x0_hand': z_hand,
        }
        return pack
    

class TextConditionedSparseStructureLatent(TextConditionedMixin, SparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    """
    pass


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass
    
class ImageConditionedSparseStructureLatentSDF(ImageConditionedMixinRotation, SparseStructureLatentSDF):
    """
    Image-conditioned sparse structure dataset
    """
    pass

class ImageConditionedSparseStructureLatentSDFConditioned(ImageConditionedMixinRotationConditioned, SparseStructureLatentSDFConditioned):
    """
    Image-conditioned sparse structure dataset with complete conditioning
    """
    pass

class ImageConditionedSparseStructureLatentSDFConditionedPrior(ImageConditionedSparseStructureLatentSDFConditioned):
    """
    P4 recursive student dataset (EVAL_GUIDANCE.md §7.19): every item carries,
    in addition to the standard single-view conditioning, a PRIOR SDF volume on
    the current view's grid — the per-voxel median of warped object SDFs from
    OTHER views of the same grasp (simulated streaming: what a ring buffer of
    earlier reconstructions would provide).

    Curriculum via `prior_source`:
      - "gt_corrupt": other views' GT SDFs, corrupted to imitate student-recon
        error (global offset = erode/dilate, smooth noise field, random blob
        deletion). No precompute needed — training can start immediately.
      - "student": precomputed frozen-student reconstructions loaded from
        `prior_student_dir/<instance>/f{view:03d}.npy` (offline pass,
        tools/precompute_student_recons.py). Views without a file are skipped;
        if none of the sampled views has one, the prior is dropped for that
        item (keep=0).

    Anti-copy is structural: the prior comes from OTHER views, so it is wrong
    exactly where the current image is informative. `prior_dropout` preserves
    the single-view mode; pose jitter on the source poses simulates rig
    calibration/tracking error.

    Returned extra keys: `prior_sdf` [1,64,64,64] float32 (all-ones when
    dropped) and `prior_keep` float scalar (1 = prior valid).
    """

    def __init__(self, roots, *,
                 prior_source: str = "gt_corrupt",
                 prior_student_dir: str = None,
                 prior_k_max: int = 7,
                 prior_dropout: float = 0.3,
                 prior_jitter_rot_deg: float = 3.0,
                 prior_jitter_trans: float = 0.02,
                 prior_jitter_scale: float = 0.03,
                 prior_corrupt_offset: float = 0.01,
                 prior_corrupt_noise: float = 0.03,
                 prior_corrupt_blob_p: float = 0.3,
                 **kwargs):
        assert prior_source in ("gt_corrupt", "student"), prior_source
        if prior_source == "student":
            assert prior_student_dir, "prior_source='student' needs prior_student_dir"
        self.prior_source = prior_source
        self.prior_student_dir = prior_student_dir
        self.prior_k_max = prior_k_max
        self.prior_dropout = prior_dropout
        self.prior_jitter_rot_deg = prior_jitter_rot_deg
        self.prior_jitter_trans = prior_jitter_trans
        self.prior_jitter_scale = prior_jitter_scale
        self.prior_corrupt_offset = prior_corrupt_offset
        self.prior_corrupt_noise = prior_corrupt_noise
        self.prior_corrupt_blob_p = prior_corrupt_blob_p
        self._prior_view_cache = {}
        super().__init__(roots, **kwargs)

    # -- helpers ------------------------------------------------------------

    def _views_with_sdf(self, inst_dir, instance_name):
        key = inst_dir
        if key not in self._prior_view_cache:
            import re
            views = []
            sdf_dir = os.path.join(inst_dir, "sdfs")
            if os.path.isdir(sdf_dir):
                for fn in os.listdir(sdf_dir):
                    m = re.match(rf"{re.escape(instance_name)}_f(\d+)__object\.npy$", fn)
                    if m:
                        views.append(int(m.group(1)))
            self._prior_view_cache[key] = sorted(views)
        return self._prior_view_cache[key]

    def _load_meta(self, inst_dir, instance_name, view):
        with open(os.path.join(inst_dir, f"{instance_name}_f{view:03d}_meta.json")) as f:
            return json.load(f)

    def _jitter_meta(self, meta):
        """Perturb a source view's pose (rig calibration/tracking noise)."""
        import copy as _copy
        m = _copy.deepcopy(meta)
        p = m["pose"]
        s = float(p["s_aug"]) * (1.0 + np.random.uniform(-self.prior_jitter_scale,
                                                         self.prior_jitter_scale))
        t = np.asarray(p["t_aug"], dtype=np.float64) + np.random.uniform(
            -self.prior_jitter_trans, self.prior_jitter_trans, size=3)
        ang = np.deg2rad(np.random.uniform(-self.prior_jitter_rot_deg,
                                           self.prior_jitter_rot_deg))
        ax = np.random.randn(3)
        ax /= (np.linalg.norm(ax) + 1e-12)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R_d = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        R = R_d @ np.asarray(p["R_fixed"], dtype=np.float64)
        p["s_aug"], p["t_aug"], p["R_fixed"] = s, t.tolist(), R.tolist()
        return m

    def _corrupt_gt(self, sdf):
        """Imitate student-reconstruction error on a GT SDF."""
        from scipy.ndimage import gaussian_filter
        out = sdf.astype(np.float32).copy()
        # global erode/dilate
        out += np.random.uniform(-self.prior_corrupt_offset, self.prior_corrupt_offset)
        # smooth noise field (unit-std after smoothing, then scaled)
        amp = np.random.uniform(0.0, self.prior_corrupt_noise)
        if amp > 0:
            n = gaussian_filter(np.random.randn(*out.shape).astype(np.float32), sigma=2.0)
            out += n * (amp / (n.std() + 1e-8))
        # random blob deletion (missed geometry)
        if np.random.rand() < self.prior_corrupt_blob_p:
            inside = np.argwhere(out < 0)
            if len(inside):
                c = inside[np.random.randint(len(inside))]
                r = np.random.uniform(4.0, 12.0)
                ii, jj, kk = np.ogrid[:out.shape[0], :out.shape[1], :out.shape[2]]
                d2 = (ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2
                out[d2 < r * r] = np.maximum(out[d2 < r * r], 0.1)
        return out

    def _load_src_sdf(self, root, inst_dir, instance_name, view):
        if self.prior_source == "gt_corrupt":
            p = os.path.join(inst_dir, "sdfs", f"{instance_name}_f{view:03d}__object.npy")
            if not os.path.exists(p):
                return None
            sdf = np.load(p)
            # snapshots stay deterministic-ish: no corruption in inference mode
            return sdf.astype(np.float32) if getattr(self, "inference", False) \
                else self._corrupt_gt(sdf)
        p = os.path.join(self.prior_student_dir, instance_name, f"f{view:03d}.npy")
        if not os.path.exists(p):
            return None
        return np.load(p).astype(np.float32)

    # -- main hook ----------------------------------------------------------

    def get_instance(self, root, instance_name):
        from ..utils.mv_warp_np import warp_sdf
        pack = super().get_instance(root, instance_name)
        view = int(pack["frame_id"])

        prior = None
        keep = 0.0
        drop = (not getattr(self, "inference", False)) and \
               (np.random.rand() < self.prior_dropout)
        if not drop:
            inst_dir = os.path.join(root, "data_pose_norm", instance_name)
            others = [v for v in self._views_with_sdf(inst_dir, instance_name)
                      if v != view]
            if others:
                k = np.random.randint(1, self.prior_k_max + 1) if not getattr(self, "inference", False) \
                    else min(self.prior_k_max, len(others))
                chosen = list(np.random.choice(others, size=min(k, len(others)),
                                               replace=False))
                meta_dst = self._load_meta(inst_dir, instance_name, view)
                warped = []
                for v in chosen:
                    sdf = self._load_src_sdf(root, inst_dir, instance_name, int(v))
                    if sdf is None:
                        continue
                    meta_src = self._load_meta(inst_dir, instance_name, int(v))
                    if not getattr(self, "inference", False):
                        meta_src = self._jitter_meta(meta_src)
                    warped.append(warp_sdf(sdf, meta_src, meta_dst))
                if warped:
                    prior = np.median(np.stack(warped), axis=0).astype(np.float32) \
                        if len(warped) > 1 else warped[0].astype(np.float32)
                    keep = 1.0

        if prior is None:
            prior = np.ones((64, 64, 64), dtype=np.float32)  # "far outside" = no info
        pack["prior_sdf"] = torch.from_numpy(prior).unsqueeze(0)
        pack["prior_keep"] = torch.tensor(keep, dtype=torch.float32)
        return pack
