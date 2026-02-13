import itertools
from typing import *
import copy
from unicodedata import name
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch
import torch.nn.functional as F

from ..basic import BasicTrainer
from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips

import copy
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree



class SLatVaeGaussianTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """

    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_kl: float = 1e-6,
        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_kl = lambda_kl
        self.regularizations = regularizations

        self._init_renderer()

    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['decoder'].rep_config['2d_filter_kernel_size']

    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0)
        return ret

    @torch.no_grad()
    def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['decoder'].resolution - 0.5
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['decoder'].rep_config['num_gaussians'], -1).reshape(-1, 3)
        status = {
            'xyz': xyz,
            'offset': offset,
            'scale': torch.cat([g.get_scaling for g in reps], dim=0),
            'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
        }

        for k in list(status.keys()):
            status[k] = {
                'mean': status[k].mean().item(),
                'max': status[k].max().item(),
                'min': status[k].min().item(),
            }

        return status

    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            terms[f'reg_vol'] = volume.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms

    def training_losses(
        self,
        feats: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = self.training_models['encoder'](feats, sample_posterior=True, return_raw=True)
        reps = self.training_models['decoder'](z)
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)

        terms = edict(loss = 0.0, rec = 0.0)

        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]

        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        status = self._get_status(z, reps)

        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items()}
            gt_images.append(args['image'] * args['alpha'][:, None])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
            reps.extend(self.models['decoder'](z))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results['color'])

        ## Concatenate views
        miltiview_images = torch.cat([
            torch.cat(miltiview_images[:2], dim=-2),
            torch.cat(miltiview_images[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'

        return ret_dict


class SLatVaeGaussianTrainerPose(BasicTrainer):
    """
    Trainer for structured latent VAE.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """

    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_kl: float = 1e-6,
        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_kl = lambda_kl
        self.regularizations = regularizations

        self._init_renderer()

    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['decoder'].rep_config['2d_filter_kernel_size']

    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0)
        return ret

    @torch.no_grad()
    def _pose_to_canonical_u2_params(self, R_row, s_aug, t_aug, c0, eps=1e-12):
        """
        Map x_pose_u2 -> x_canon_u2
        
        Forward you used when generating posed grids:
        x_pose = ((x_canon - c0) @ R) * s_aug + t_aug
        
        Inverse:
        x_canon = ((x_pose - t_aug)/s_aug) @ R^T + c0
             = (x_pose @ R^T) * (1/s_aug) + (c0 - (t_aug @ R^T)*(1/s_aug))
        """
        B = R_row.shape[0]
        s_aug = s_aug.view(B, 1).clamp_min(eps)   # (B,1)
        k = 1.0 / s_aug                           # (B,1)
        
        tR = torch.einsum("bi,bij->bj", t_aug, R_row.transpose(1, 2))  # t @ R^T  -> (B,3)
        b = c0 - tR * k                                                 # (B,3)
        return R_row, k, b

    @torch.no_grad()
    def _pose_to_render_params(self, R_row, s_aug, t_aug, c0, *, target="unit2", eps=1e-12):
        """
        Maps pose-space coords back to the *render* canonical space.
        
        Pose generation was:
        x_pose = ((x_can - c0) @ R) * s + t     (row-vector convention)
        
        Inverse (back to canonical render space):
        x_can = (x_pose @ R^T) * (1/s) + (c0 - (t @ R^T)/s)
        
        If your render space is unitcube [-0.5,0.5]^3, additionally do x_uc = 0.5 * x_can.
        """
        B = R_row.shape[0]
        s_aug = s_aug.view(B, 1).clamp_min(eps)
        
        k = 1.0 / s_aug                                   # (B,1)
        tR = torch.einsum("bi,bij->bj", t_aug, R_row.transpose(1, 2))  # t @ R^T, (B,3)
        b = c0 - tR * k                                    # (B,3)
    
        if target == "unitcube":
            k = 0.5 * k
            b = 0.5 * b
            
        return R_row, k, b


    def xyz_stats(self,xyz01: torch.Tensor, name="xyz01"):
        # xyz01: [N,3]
        x = xyz01.detach().float().reshape(-1, 3)

        # percentiles per axis
        qs = torch.quantile(x, torch.tensor([0.01, 0.50, 0.99], device=x.device), dim=0)
        p01, p50, p99 = qs[0], qs[1], qs[2]

        mn = x.min(dim=0).values
        mx = x.max(dim=0).values
        mean = x.mean(dim=0)

        # out of [0,1]
        below = (x < 0).any(dim=1)
        above = (x > 1).any(dim=1)
        oob = below | above

        # how far out (L_infty distance to box)
        # violation per point: max( -x, x-1, 0 ) across coords
        viol = torch.maximum(torch.maximum(-x, x - 1.0), torch.zeros_like(x))
        viol_linf = viol.max(dim=1).values  # [N]
        max_viol = viol_linf.max().item()
        p99_viol = torch.quantile(viol_linf, 0.99).item()

        return {
            f"{name}/N": x.shape[0],
            f"{name}/mean": mean.cpu().tolist(),
            f"{name}/min": mn.cpu().tolist(),
            f"{name}/max": mx.cpu().tolist(),
            f"{name}/p01": p01.cpu().tolist(),
            f"{name}/p50": p50.cpu().tolist(),
            f"{name}/p99": p99.cpu().tolist(),
            f"{name}/oob_frac": oob.float().mean().item(),
            f"{name}/below_frac": below.float().mean().item(),
            f"{name}/above_frac": above.float().mean().item(),
            f"{name}/max_violation": max_viol,
            f"{name}/p99_violation": p99_viol,
        }


    def _make_unit2_aabb(self, device, dtype):
        # aabb = [min_x, min_y, min_z, size_x, size_y, size_z]
        return torch.tensor([-1.0, -1.0, -1.0, 2.0, 2.0, 2.0], device=device, dtype=dtype)

    def _transform_reps_for_loss(
        self,
        reps,  # list[Gaussian], len B
        *,
        R_row, s_aug, t_aug, c0,
        s_norm, c_norm,
        origin=None, voxel_size=None,
            blender_scale=None, blender_offset=None,
        intrinsics=None, extrinsics=None,
        posed_mask=None,                    # [B] bool or None
        force_unit2_aabb: bool = True,
        debug: bool = False,
    ):
        """
        Transform decoder reps from POSE -> WORLD.

        Assumes:
        rep.get_xyz returns "pose-space/world-space xyz" depending on rep.aabb mapping.
        We treat rep.get_xyz as the POSE-space xyz for the pose transform.

        Returns:
        reps_world: list[Gaussian] len B, shallow-copied objects with transformed xyz (+ optional aabb).
        """
        B = len(reps)
        assert R_row.shape[0] == B
        assert s_aug.shape[0] == B
        assert t_aug.shape[0] == B
        assert c0.shape[0] == B
        assert s_norm.shape[0] == B
        assert c_norm.shape[0] == B

        if posed_mask is not None:
            # make it a bool tensor on CPU/GPU doesn’t matter for indexing, but keep simple
            posed_mask = posed_mask.to(device=R_row.device).bool()
            assert posed_mask.shape[0] == B

        # This must be the *correct* one you validated with the cost-matrix diag:
        # xyz_world = (xyz_pose @ R^T) * k + b
        R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm, blender_scale, blender_offset)
        #R, k, b = self._pose_to_render_params(R_row, s_aug, t_aug, c0, target="unit2")  # or "unitcube"
        #R, k, b = self._pose_to_canonical_u2_params(R_row, s_aug, t_aug, c0)

        # R: [B,3,3], k: [B] or [B,1], b: [B,3]

        reps_world = []

        for i in range(B):
            rep_src = reps[i]

            # If sample is not posed, pass through unchanged
            if posed_mask is not None and (not bool(posed_mask[i].item())):
                reps_world.append(rep_src)
                continue

            # shallow copy is safe; tensors are still shared unless we replace them
            rep_w = copy.copy(rep_src)

            # read pose-space xyz (world units, from aabb mapping)
            xyz_pose = rep_src.get_xyz.view(-1, 3)

            Ri = R[i]                       # [3,3]
            ki = k[i].reshape(1, 1)         # [1,1]
            bi = b[i].reshape(1, 3)         # [1,3]

            # pose -> world
            xyz_world = (xyz_pose @ Ri.t()) * ki + bi

            # enforce unit2 aabb so rep_w.get_xyz == xyz_world
            if force_unit2_aabb:
                aabb_u2 = self._make_unit2_aabb(device=xyz_world.device, dtype=rep_src.aabb.dtype if torch.is_tensor(rep_src.aabb) else xyz_world.dtype)
                rep_w.aabb = aabb_u2

                # world [-1,1] -> internal [0,1]
                xyz01 = (xyz_world + 1) *0.5
            else:
                # map into existing aabb
                aabb = rep_w.aabb.to(device=xyz_world.device, dtype=xyz_world.dtype)
                aabb_min  = aabb[:3].view(1, 3)
                aabb_size = aabb[3:].view(1, 3).clamp_min(1e-12)
                xyz01 = (xyz_world - aabb_min) / aabb_size

            # IMPORTANT: assign, don't in-place copy_ into an existing tensor
            rep_w._xyz = xyz01.reshape_as(rep_src._xyz).to(dtype=rep_src._xyz.dtype)

            # Optional: if your Gaussian uses per-point scale/rotation in same frame,
            # you should also transform them. Uncomment only if these attrs exist.
            #
            # if hasattr(rep_src, "_scaling"):
            #     rep_w._scaling = rep_src._scaling * ki.to(rep_src._scaling.dtype)
            # if hasattr(rep_src, "_rotation"):
            #     # depends on rotation parameterization (quat/matrix). Need your exact convention.
            #     pass

            if debug:
                xyz_back = rep_w.get_xyz.view(-1, 3)
                err = (xyz_back - xyz_world).abs()
                print(f"[i={i}] xyz_consist max={err.max().item():.3e} mean={err.mean().item():.3e}")

            reps_world.append(rep_w)

        return reps_world

    def _quat_from_rotmat(self, R: torch.Tensor) -> torch.Tensor:
        """
        R: (3,3) or (...,3,3) rotation matrix
        Returns q: (...,4) quaternion (w,x,y,z)
        """
        # robust conversion
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

        m = t > 0
        if m.any():
            tt = t[m]
            r = torch.sqrt(1.0 + tt)
            qw = 0.5 * r
            r = 0.5 / r
            qx = (R[m, 2, 1] - R[m, 1, 2]) * r
            qy = (R[m, 0, 2] - R[m, 2, 0]) * r
            qz = (R[m, 1, 0] - R[m, 0, 1]) * r
            q[m] = torch.stack([qw, qx, qy, qz], dim=-1)

        m = ~m
        if m.any():
            Rm = R[m]
            i = torch.argmax(torch.stack([Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]], dim=1), dim=1)
            q_m = torch.zeros((Rm.shape[0], 4), device=R.device, dtype=R.dtype)

            for idx in [0, 1, 2]:
                mm = i == idx
                if not mm.any():
                    continue
                r = torch.sqrt(
                    1.0 + Rm[mm, idx, idx]
                    - Rm[mm, (idx + 1) % 3, (idx + 1) % 3]
                    - Rm[mm, (idx + 2) % 3, (idx + 2) % 3]
                )
                qq = torch.zeros((mm.sum(), 4), device=R.device, dtype=R.dtype)
                qq[:, 1 + idx] = 0.5 * r
                r = 0.5 / r
                qq[:, 0] = (Rm[mm, (idx + 2) % 3, (idx + 1) % 3] - Rm[mm, (idx + 1) % 3, (idx + 2) % 3]) * r
                qq[:, 1 + (idx + 1) % 3] = (Rm[mm, (idx + 1) % 3, idx] + Rm[mm, idx, (idx + 1) % 3]) * r
                qq[:, 1 + (idx + 2) % 3] = (Rm[mm, (idx + 2) % 3, idx] + Rm[mm, idx, (idx + 2) % 3]) * r
                q_m[mm] = qq

            q[m] = q_m

        return F.normalize(q, dim=-1)

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product; q=(w,x,y,z). Shapes (...,4)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    def _inv_scaling_activation(self, s: torch.Tensor, act: str, eps: float = 1e-8) -> torch.Tensor:
        """Inverse of activation used for scaling."""
        s = torch.clamp(s, min=eps)
        if act in ["exp", "exponential"]:
            return torch.log(s)
        if act in ["softplus"]:
            # softplus(x)=log(1+exp(x)) -> x=log(exp(s)-1)
            return torch.log(torch.expm1(s) + eps)
        if act in ["relu"]:
            # not really invertible; best effort
            return s
        # if unknown, do nothing (better than corrupting)
        return s


    @torch.no_grad()
    def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm, blender_scale=None, blender_offset=None, eps=1e-12):
        """
        Inputs:
        R_row : (B,3,3)
        s_aug : (B,) or (B,1)
        t_aug : (B,3)
        c0    : (B,3)
        s_norm: (B,) or (B,1)
        c_norm: (B,3)

        Returns:
        R: (B,3,3)     (same as R_row; used as xyz_pose @ R^T)
        k: (B,1)       scalar per sample
        b: (B,3)
        """
        # ensure shapes
        B = R_row.shape[0]
        assert R_row.shape == (B,3,3)
        assert t_aug.shape == (B,3)
        assert c0.shape == (B,3)
        assert c_norm.shape == (B,3)

        s_aug  = s_aug.view(B, 1)
        s_norm = s_norm.view(B, 1)

        k = 1.0 / (s_aug.clamp_min(eps) * s_norm.clamp_min(eps))     # (B,1)

        # (t @ R^T): row-vector multiply -> (B,3)
        tR = torch.einsum("bi,bij->bj", t_aug, R_row.transpose(1,2))

        b = c_norm + (c0 / s_norm.clamp_min(eps)) - (tR * k)         # (B,3)

        if blender_scale is not None:
            sb = blender_scale.view(B, 1).to(k.device)
            k = k * sb
            
            b = b * sb
        
        if blender_offset is not None:
            ob = blender_offset.view(B, 3).to(b.device)
            b = b + ob
            R = R_row
        return R_row, k, b


    @torch.no_grad()
    def _get_status(self, z, reps):
        status = {}

        # --- XYZ ---
        xyz_list = []
        for g in reps:
            try:
                xyz_list.append(g.get_xyz.reshape(-1, 3))
            except Exception:
                pass
        if len(xyz_list) > 0:
            xyz = torch.cat(xyz_list, dim=0)
            # bbox diag as a single health scalar
            mn = xyz.min(dim=0).values
            mx = xyz.max(dim=0).values
            diag = (mx - mn).norm()
            status["xyz/diag"] = float(diag)
            status.update(self._stats_tensor(xyz[:, 0], "xyz/x"))
            status.update(self._stats_tensor(xyz[:, 1], "xyz/y"))
            status.update(self._stats_tensor(xyz[:, 2], "xyz/z"))

        # --- Scaling ---
        sc_list = []
        for g in reps:
            try:
                sc_list.append(g.get_scaling.reshape(-1, 3))
            except Exception:
                pass
        if len(sc_list) > 0:
            scales = torch.cat(sc_list, dim=0)  # (N,3)
            status.update(self._stats_tensor(scales[:, 0], "scale/x"))
            status.update(self._stats_tensor(scales[:, 1], "scale/y"))
            status.update(self._stats_tensor(scales[:, 2], "scale/z"))
            vol = scales[:, 0] * scales[:, 1] * scales[:, 2]
            status.update(self._stats_tensor(vol, "scale/vol"))

        # --- Opacity ---
        op_list = []
        for g in reps:
            try:
                op_list.append(g.get_opacity.reshape(-1))
            except Exception:
                pass
        if len(op_list) > 0:
            opacity = torch.cat(op_list, dim=0)
            status.update(self._stats_tensor(opacity, "opacity"))

        return status


    @torch.no_grad()
    def _stats_tensor(self, x: torch.Tensor, name: str, sample_max: int = 200000):
        x = x.detach().float().reshape(-1)
        if x.numel() == 0:
            return {
                f"{name}/mean": 0.0, f"{name}/min": 0.0, f"{name}/max": 0.0,
                f"{name}/p10": 0.0, f"{name}/p50": 0.0, f"{name}/p90": 0.0,
            }

        if x.numel() > sample_max:
            idx = torch.randint(0, x.numel(), (sample_max,), device=x.device)
            x = x[idx]

        qs = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
        return {
            f"{name}/mean": float(x.mean()),
            f"{name}/min":  float(x.min()),
            f"{name}/max":  float(x.max()),
            f"{name}/p10":  float(qs[0]),
            f"{name}/p50":  float(qs[1]),
            f"{name}/p90":  float(qs[2]),
        }


    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            terms[f'reg_vol'] = volume.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms
    def get_weight(self, rep):
        # Try a few common fields; adapt if your rep has a different name
        if hasattr(rep, "_opacity") and rep._opacity is not None:
            w = rep._opacity
            # if stored as logits, you might want sigmoid:
            # w = torch.sigmoid(w)
            return w.detach().float().reshape(-1)
        if hasattr(rep, "_density") and rep._density is not None:
            w = rep._density
            # if stored as logits, apply activation if needed:
            # w = F.softplus(w)
            return w.detach().float().reshape(-1)
        return None

    def scaling_stats(self, rep, name="scaling", sample=200000):
        """
        Report stats for the *effective/physical* Gaussian scales actually used by the renderer,
        i.e. rep.get_scaling (includes bias + min kernel size).
        """
        if getattr(rep, "_scaling", None) is None:
            return None

        # This is the correct "physical" scaling used downstream
        s = rep.get_scaling.detach().float()   # [N,3]

        # subsample by number of points (not numel)
        if s.shape[0] > sample:
            idx = torch.randint(0, s.shape[0], (sample,), device=s.device)
            s = s[idx]

        mn = s.min(0).values
        mx = s.max(0).values
        qs = torch.quantile(
            s,
            torch.tensor([0.01, 0.5, 0.99], device=s.device),
            dim=0
        )

        return {
            f"{name}/min": mn.cpu().tolist(),
            f"{name}/p01": qs[0].cpu().tolist(),
            f"{name}/p50": qs[1].cpu().tolist(),
            f"{name}/p99": qs[2].cpu().tolist(),
            f"{name}/max": mx.cpu().tolist(),
        }



    def training_losses(
        self,
        feats: SparseTensor,
        image: torch.Tensor,          # [B,3,H,W]
        alpha: torch.Tensor,          # [B,H,W] or [B,1,H,W]
        extrinsics: torch.Tensor,     # [B,4,4]
        intrinsics: torch.Tensor,     # [B,3,3]
        R_row: torch.Tensor,          # [B,3,3]
        s_aug: torch.Tensor,          # [B]
        t_aug: torch.Tensor,          # [B,3]
        c0: torch.Tensor,             # [B,3]
        s_norm: torch.Tensor,         # [B]
        c_norm: torch.Tensor,         # [B,3]
        origin: torch.Tensor = None,
        voxel_size: torch.Tensor = None,
        blender_scale: torch.Tensor = None,
        blender_offset: torch.Tensor = None,
        posed_mask: torch.Tensor = None,  # [B] bool
        return_aux: bool = False,
        **kwargs
    ):
        # -------------------------
        # Encode / decode (POSE-space reps)
        # -------------------------
        z, mean, logvar = self.training_models["encoder"](
            feats, sample_posterior=True, return_raw=True
        )
        reps_pose = self.training_models["decoder"](z)   # list[Gaussian], len==B

        # -------------------------
        # Pose -> World transform (only if posed)
        # -------------------------
        reps_world = self._transform_reps_for_loss(
            reps_pose,
            R_row=R_row, s_aug=s_aug, t_aug=t_aug, c0=c0,
            s_norm=s_norm, c_norm=c_norm,
            origin=origin, voxel_size=voxel_size,
            blender_scale=blender_scale, blender_offset=blender_offset,
            posed_mask=posed_mask,
            # keep debug False in training
            debug=False,
        )
        assert reps_world[0]._xyz.requires_grad == reps_pose[0]._xyz.requires_grad

        # -------------------------
        # Render in WORLD
        # -------------------------
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps_world, extrinsics, intrinsics)

        rec_image = render_results["color"]  # [B,3,H,W]

        # bg_color handling (robust)
        if "bg_color" in render_results and render_results["bg_color"] is not None:
            bg = render_results["bg_color"]  # usually [B,3]
            while bg.dim() < 4:
                bg = bg[..., None]
            # bg now [B,3,1,1]
        else:
            bg = torch.zeros_like(rec_image)

        # alpha shape normalize
        if alpha.dim() == 3:          # [B,H,W] -> [B,1,H,W]
            a = alpha[:, None]
        elif alpha.dim() == 4:        # already [B,1,H,W]
            a = alpha
        else:
            raise ValueError(f"alpha must be [B,H,W] or [B,1,H,W], got {alpha.shape}")

        # GT composite on same background
        gt_image = image * a + (1.0 - a) * bg

        # -------------------------
        # Loss terms
        # -------------------------
        terms = edict(loss=0.0, rec=0.0)

        if self.loss_type == "l1":
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == "l2":
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if self.lambda_ssim > 0:
            terms["ssim"] = 1.0 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]

        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]

        terms["loss"] = terms["loss"] + terms["rec"]

        # KL
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1.0)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        # regularization on WORLD reps (important if reg depends on xyz)
        reg_loss, reg_terms = self._get_regularization_loss(reps_world)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        # status: prefer world reps if status uses xyz/extent
        status = self._get_status(z, reps_world)

        if return_aux:
            return terms, status, {"rec_image": rec_image, "gt_image": gt_image}

        return terms, status


    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        import copy
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        # -------------------------
        # inference
        # -------------------------
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []

        # pose metadata (needed to map POSE->WORLD)
        pose_pack = {
            "R_row": [], "s_aug": [], "t_aug": [], "c0": [], "s_norm": [], "c_norm": [],
            "origin": [], "voxel_size": [], "blender_scale": [], "blender_offset":[],
        }

        reps_pose = []
        num_samples = 4
        it = iter(dataloader)
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(it)
            
            args = {k: v[:batch].cuda() for k, v in data.items()}

            # GT for visualization
            gt_images.append(args["image"] * args["alpha"][:, None])

            # Cameras are WORLD cameras
            exts.append(args["extrinsics"])
            ints.append(args["intrinsics"])
            
            # Pose params (must match the samples in this batch)
            for k in pose_pack:
                if k in args:
                    pose_pack[k].append(args[k])
                else:
                    # Some datasets might not have origin/voxel_size; keep them optional
                    pose_pack[k].append(None)

            # Encode/decode
            z = self.models["encoder"](args["feats"], sample_posterior=True, return_raw=False)
            reps_pose.extend(self.models["decoder"](z))

        gt_images = torch.cat(gt_images, dim=0)
        ret_dict["gt_image"] = {"value": gt_images, "type": "image"}

        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)

        # Concatenate pose params (skip optional None fields)
        pose_args = {}
        for k, chunks in pose_pack.items():
            if all(c is None for c in chunks):
                pose_args[k] = None
            else:
                # if some are None and some are tensors -> that's an error
                assert not any(c is None for c in chunks), f"Missing {k} for some batches"
                pose_args[k] = torch.cat(chunks, dim=0)

        # -------------------------
        # POSE -> WORLD reps (critical change)
        # -------------------------
        reps_world = self._transform_reps_for_loss(
            reps_pose,
            R_row=pose_args["R_row"],
            s_aug=pose_args["s_aug"],
            t_aug=pose_args["t_aug"],
            c0=pose_args["c0"],
            s_norm=pose_args["s_norm"],
            c_norm=pose_args["c_norm"],
            origin=pose_args.get("origin", None),
            voxel_size=pose_args.get("voxel_size", None),
            blender_scale=pose_args.get("blender_scale", None),
            blender_offset=pose_args.get("blender_offset", None),
            intrinsics=ints,
            extrinsics=exts,
            debug=False,
        )

        # -------------------------
        # render single view (GT cameras)
        # -------------------------
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps_world, exts, ints)
        ret_dict["rec_image"] = {"value": render_results["color"], "type": "image"}

        # -------------------------
        # render multiview (synthetic cameras, WORLD frame)
        # -------------------------
        self.renderer.rendering_options.resolution = 512

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        multiview_images = []
        for yaw, pit in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pit),
                np.cos(yaw) * np.cos(pit),
                np.sin(pit),
            ], device="cuda", dtype=torch.float32) * 2

            fov = torch.deg2rad(torch.tensor(30.0, device="cuda"))
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0], device="cuda", dtype=torch.float32),
                torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32),
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)

            mv = self._render_batch(reps_world, extrinsics, intrinsics)
            multiview_images.append(mv["color"])

        multiview_images = torch.cat([
            torch.cat(multiview_images[:2], dim=-2),
            torch.cat(multiview_images[2:], dim=-2),
        ], dim=-1)

        ret_dict["miltiview_image"] = {"value": multiview_images, "type": "image"}

        self.renderer.rendering_options.bg_color = "random"
        return ret_dict
