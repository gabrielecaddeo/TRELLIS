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

    # def _transform_reps_for_loss(
    #     self,
    #     reps,
    #     *,
    #     R_row, s_aug, t_aug, c0, s_norm, c_norm,
    #     origin, voxel_size,
    #     posed_mask=None
    # ):
    #     """
    #     reps[i]._xyz is in normalized [0,1] coordinates (AABB space).

    #     We want to render in WORLD space (same frame as extrinsics/intrinsics),
    #     but the renderer expects rep._xyz to stay in AABB-normalized coords.

    #     So we:
    #     1) interpret current xyz01 as 'posed grid normalized' (0..1 over res)
    #     2) map xyz01 -> posed coords using (origin, voxel_size, res)
    #     3) posed -> world using (R_row, s_aug, t_aug, c0, s_norm, c_norm)
    #     4) world -> aabb-normalized coords and write back to rep._xyz

    #     Scaling is updated consistently in activated (softplus) space and mapped back to raw.
    #     """
    #     import copy
    #     import torch
    #     import torch.nn.functional as F

    #     eps = 1e-8
    #     B = len(reps)
    #     device = reps[0]._xyz.device
    #     dtype  = reps[0]._xyz.dtype

    #     # --- move inputs to device ---
    #     R_row  = R_row.to(device=device, dtype=dtype)                 # (B,3,3)
    #     s_aug  = s_aug.to(device=device, dtype=dtype).view(-1)        # (B,)
    #     t_aug  = t_aug.to(device=device, dtype=dtype)                 # (B,3)
    #     c0     = c0.to(device=device, dtype=dtype)                    # (B,3)
    #     s_norm = s_norm.to(device=device, dtype=dtype).view(-1)       # (B,)
    #     c_norm = c_norm.to(device=device, dtype=dtype)                # (B,3)

    #     origin     = origin.to(device=device, dtype=dtype)            # (B,3)  posed grid origin (unit2)
    #     voxel_size = voxel_size.to(device=device, dtype=dtype).view(-1)  # (B,) posed voxel size (unit2)

    #     if posed_mask is not None:
    #         posed_mask = posed_mask.to(device=device)

    #     # posed -> world params: x_w = (x_pose @ R.T) * k + b
    #     # IMPORTANT: do NOT wrap the whole transform in no_grad (we need grads wrt decoder outputs).
    #     R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)
    #     R = R.to(device=device, dtype=dtype)   # (B,3,3)
    #     k = k.to(device=device, dtype=dtype)   # (B,1,1)
    #     b = b.to(device=device, dtype=dtype)   # (B,1,3)

    #     # decoder resolution
    #     res = float(self.models["decoder"].resolution)
    #     grid_scale = (voxel_size.view(B, 1, 1) * res)  # (B,1,1) so xyz01*(res*vox)=unit2-length

    #     # ---- softplus scaling helpers (assume always softplus) ----
    #     bias_default = float(self.models["decoder"].rep_config.get("scaling_bias", 0.0))

    #     def softplus_inv(y):
    #         # inverse of softplus(x)=log(1+exp(x)) (approx; consistent with F.softplus for beta=1)
    #         y = torch.clamp(y, min=eps)
    #         return torch.log(torch.expm1(y) + eps)

    #     out = []
    #     for i in range(B):
    #         rep2 = copy.copy(reps[i])

    #         if posed_mask is not None and not bool(posed_mask[i].item()):
    #             out.append(rep2)
    #             continue

    #         Ri  = R[i]                         # (3,3)
    #         ki  = k[i]                         # (1,1)
    #         bi  = b[i]                         # (1,3)
    #         ori = origin[i].view(1, 3)         # (1,3)
    #         gsi = grid_scale[i]                # (1,1)

    #         # --------- (A) xyz: xyz01 -> posed -> world ----------
    #         xyz01 = rep2._xyz.view(-1, 3)                 # (Ng,3) in [0,1] (grid-normalized)
    #         xyz_pose = ori + xyz01 * gsi                  # (Ng,3) posed unit2 coords
    #         xyz_w = (xyz_pose @ Ri.t()) * ki + bi         # (Ng,3) world coords
    #         mn = xyz_w.min(0).values
    #         mx = xyz_w.max(0).values
    #         print("xyz_w min/max:", mn, mx)
    #         # xyz_w = 0.5*xyz_w
    #         # --------- (B) world -> rep AABB-normalized coords ----------
    #         # The renderer expects _xyz in AABB-normalized coords.
    #         # In this codebase Gaussian.aabb is typically [min_x,min_y,min_z, size_x,size_y,size_z].
    #         if hasattr(rep2, "aabb") and rep2.aabb is not None:
    #             aabb = rep2.aabb
    #             if not torch.is_tensor(aabb):
    #                 aabb = torch.tensor(aabb, device=device, dtype=dtype)
    #             aabb = aabb.to(device=device, dtype=dtype)
    #             aabb_min  = aabb[:3].view(1, 3)
    #             aabb_size = aabb[3:].view(1, 3)
    #         else:
    #             # fallback consistent with your status code (-0.5..0.5 box)
    #             aabb_min  = torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=dtype).view(1, 3)
    #             aabb_size = torch.tensor([ 2.0,  2.0,  2.0], device=torch.device, dtype=dtype).view(1, 3)


    #         xyz01_w = (xyz_w - aabb_min) / torch.clamp(aabb_size, min=eps)
    #         rep2._xyz = xyz01_w
    #         xyz_render = aabb_min + rep2._xyz * aabb_size
    #         print("xyz_render min/max:", xyz_render.min(0).values, xyz_render.max(0).values)
    #         # --------- (C) rotation (quat) ----------
    #         q = rep2._rotation
    #         if q is not None and q.shape[-1] == 4:
    #             qR = self._quat_from_rotmat(Ri).view(1, 4).expand_as(q)
    #             rep2._rotation = F.normalize(self._quat_mul(qR, q), dim=-1)

    #         # --------- (D) scaling ----------
    #         # Keep scaling consistent with xyz mapping.
    #         # xyz01 -> posed multiplies lengths by gsi (scalar)
    #         # posed -> world multiplies by ki (scalar)
    #         # world -> xyz01_w divides by aabb_size (per-axis); we use mean(aabb_size) since ki is uniform.
    #         # (If aabb_size is [1,1,1], this reduces to gsi*ki.)
    #         # aabb_s = aabb_size.mean().clamp_min(eps)  # scalar
    #         aabb_size_safe = aabb_size.clamp_min(eps)          # (1,3)

    #         # gsi and ki are scalars for sample i
    #         sf_scalar = (gsi.squeeze() * ki.squeeze()).clamp_min(eps)  # scalar

    #         # per-axis factor, broadcasts to (Ng,3) automatically
    #         scale_factor = sf_scalar / aabb_size_safe    
    #         # Gaussian stores raw scaling in rep2._scaling; get_scaling applies softplus(raw + bias)
    #         raw = rep2._scaling
    #         if raw is not None:
    #             bias_val = getattr(rep2, "scaling_bias", bias_default)
    #             if not torch.is_tensor(bias_val):
    #                 bias_val = torch.tensor(bias_val, device=device, dtype=dtype)

    #             pre = raw + bias_val
    #             s_act = F.softplus(pre)                         # activated
    #             s_new = s_act * scale_factor                    # rescale widths
    #             pre_new = softplus_inv(s_new)
    #             rep2._scaling = pre_new - bias_val

    #         out.append(rep2)

    #     return out

    ## Before the gold
    # def _transform_reps_for_loss(
    #     self,
    #     reps,
    #     *,
    #     R_row, s_aug, t_aug, c0, s_norm, c_norm,
    #     origin, voxel_size, intrinsics, extrinsics,
    #     posed_mask=None
    # ):
    #     """
    #     Transform reps from posed coords to the world frame used by camera extrinsics.

    #     Rules:
    #     - Do NOT mutate original `reps`.
    #     - Transform in PHYSICAL space:
    #         xyz      : rep.get_xyz      -> transform -> rep2.from_xyz()
    #         rotation : rep.get_rotation -> compose  -> rep2.from_rotation()
    #         scaling  : rep.get_scaling  -> scale    -> rep2.from_scaling()
    #     """
    #     import torch
    #     import torch.nn.functional as F
    #     from ...representations import Gaussian  # adjust if needed

    #     eps = 1e-8
    #     B = len(reps)
    #     device = reps[0]._xyz.device
    #     dtype  = reps[0]._xyz.dtype

    #     # --- move inputs to device ---
    #     R_row  = R_row.to(device=device, dtype=dtype)
    #     s_aug  = s_aug.to(device=device, dtype=dtype).view(-1)
    #     t_aug  = t_aug.to(device=device, dtype=dtype)
    #     c0     = c0.to(device=device, dtype=dtype)
    #     s_norm = s_norm.to(device=device, dtype=dtype).view(-1)
    #     c_norm = c_norm.to(device=device, dtype=dtype)

    #     origin     = origin.to(device=device, dtype=dtype)
    #     voxel_size = voxel_size.to(device=device, dtype=dtype).view(-1)

    #     if posed_mask is not None:
    #         posed_mask = posed_mask.to(device=device)

    #     # posed -> world params: xyz_w = (xyz_pose @ R^T) * k + b
    #     R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)
    #     R = R.to(device=device, dtype=dtype)   # (B,3,3)
    #     k = k.to(device=device, dtype=dtype)   # (B,1,1)
    #     b = b.to(device=device, dtype=dtype)   # (B,1,3)

    #     out = []
    #     for i in range(B):
    #         print(f"[xform i={i}] s_aug={float(s_aug[i])}  t_aug={t_aug[i].tolist()}  "
    #         f"s_norm={float(s_norm[i])}  c_norm={c_norm[i].tolist()}")
    #         # also print a small checksum of R_row to ensure it's not always the same:
    #         print(f"[xform i={i}] R_row[0,:]={R_row[i,0].tolist()}")
    #         print(f"R{i}=\n{R[i]}")
    #         print(f"k{i}=\n{k[i]}")
    #         print(f"b{i}=\n{b[i]}")
    #         rep = reps[i]

    #         if posed_mask is not None and not bool(posed_mask[i].item()):
    #             out.append(rep)
    #             continue

    #         Ri = R[i]                 # (3,3)
    #         ki = k[i].view(1)         # scalar tensor
    #         bi = b[i]                 # (1,3)
    #         # print("k (scale factor) =", float(ki))

    #         # fresh container
    #         init = dict(rep.init_params)
    #         rep2 = Gaussian(device=str(rep.device), **init)

    #         # copy tensors that don't change
    #         rep2._features_dc   = rep._features_dc
    #         rep2._features_rest = rep._features_rest
    #         rep2._opacity       = rep._opacity

    #         # (A) xyz
    #         xyz_pose = rep.get_xyz.view(-1, 3)                 # (N,3) posed coords
    #         print("xyz_pose min/max:", xyz_pose.min(), xyz_pose.max())
    #         xyz_w    = (xyz_pose @ Ri.t()) * ki + bi           # (N,3) world coords
    #         if i == 0:
    #             with torch.no_grad():
    #                 # subsample
    #                 N = xyz_pose.shape[0]
    #                 m = min(N, 5000)
    #                 idx = torch.randint(0, N, (m,), device=device)
    #                 xp = xyz_pose[idx]   # pose points
    #                 xw = xyz_w[idx]      # world points computed by formula

    #                 # Build T_pose_to_world in COLUMN convention
    #                 T = torch.eye(4, device=device, dtype=dtype)
    #                 T[:3, :3] = Ri * ki.squeeze()
    #                 T[:3, 3]  = bi.view(-1)

    #                 # sanity: xw == T*xp
    #                 xw_T = self.apply_T_col(xp, T)
    #                 errX = (xw - xw_T).abs().max().item()
    #                 print(f"[UV DEBUG] max |xw - T*xp| = {errX:.3e}")

    #                 # Get camera matrices for this sample (you must pass them to this fn or read from outer scope)
    #                 # E_world: world->cam, K: intrinsics
    #                 Ew = extrinsics[i]     # (4,4)
    #                 K  = intrinsics[i]     # (3,3)

    #                 # Mode1 UV: project(world, Ew)
    #                 uv1 = utils3d.torch.project_cv(xw, Ew, K)[0]

    #                 # Try compositions for "pose points + composed extrinsics"
    #                 # A) x_cam = Ew * (T * x_pose)   => Epose = Ew @ T
    #                 uvA = utils3d.torch.project_cv(xp, Ew @ T, K)[0]

    #                 # B) if project_cv uses row-vectors internally, sometimes order flips
    #                 uvB = utils3d.torch.project_cv(xp, T @ Ew, K)[0]

    #                 # C) maybe you built inverse direction by mistake
    #                 Tinv = torch.linalg.inv(T)
    #                 uvC = utils3d.torch.project_cv(xp, Ew @ Tinv, K)[0]
    #                 uvD = utils3d.torch.project_cv(xp, Tinv @ Ew, K)[0]

    #                 def uv_err(uvX):
    #                     d = (uv1 - uvX).abs()
    #                     return float(d.max().item()), float(d.mean().item())

    #                 eA = uv_err(uvA)
    #                 eB = uv_err(uvB)
    #                 eC = uv_err(uvC)
    #                 eD = uv_err(uvD)
    #                 print(f"[UV DEBUG] err A(Ew@T)    max={eA[0]:.3e} mean={eA[1]:.3e}")
    #                 print(f"[UV DEBUG] err B(T@Ew)    max={eB[0]:.3e} mean={eB[1]:.3e}")
    #                 print(f"[UV DEBUG] err C(Ew@Tinv) max={eC[0]:.3e} mean={eC[1]:.3e}")
    #                 print(f"[UV DEBUG] err D(Tinv@Ew) max={eD[0]:.3e} mean={eD[1]:.3e}")
    #             def cam_from_E_col(xyz, E):
    #                 # column convention: x_cam = (E @ [x,1])[:3]
    #                 ones = torch.ones((xyz.shape[0],1), device=xyz.device, dtype=xyz.dtype)
    #                 Xh = torch.cat([xyz, ones], dim=1)              # [N,4]
    #                 Yh = (E @ Xh.t()).t()                           # [N,4]
    #                 return Yh[:, :3]

    #             def cam_from_E_row(xyz, E):
    #                 # row convention: x_cam = ([x,1] @ E^T)[:3]
    #                 ones = torch.ones((xyz.shape[0],1), device=xyz.device, dtype=xyz.dtype)
    #                 Xh = torch.cat([xyz, ones], dim=1)              # [N,4]
    #                 Yh = (Xh @ E.t())                               # [N,4]
    #                 return Yh[:, :3]

    #             with torch.no_grad():
    #                 # choose subset
    #                 N = xyz_pose.shape[0]
    #                 if N > 20000:
    #                     idx = torch.randint(0, N, (20000,), device=device)
    #                     xp = xyz_pose[idx]
    #                     xw = xyz_w[idx]
    #                 else:
    #                     xp, xw = xyz_pose, xyz_w

    #                 Ew = extrinsics[i]                 # world->cam, same tensor you use for rendering
    #                 Epose = Ew @ T                     # your Mode3 extrinsics

    #                 # Camera coords two ways
    #                 cam1_col = cam_from_E_col(xw, Ew)
    #                 cam3_col = cam_from_E_col(xp, Epose)
    #                 cam1_row = cam_from_E_row(xw, Ew)
    #                 cam3_row = cam_from_E_row(xp, Epose)

    #                 # Depth masks
    #                 m_col = (cam1_col[:,2] > 1e-4) & (cam3_col[:,2] > 1e-4)
    #                 m_row = (cam1_row[:,2] > 1e-4) & (cam3_row[:,2] > 1e-4)

    #                 # Errors
    #                 err_col = (cam1_col[m_col] - cam3_col[m_col]).abs()
    #                 err_row = (cam1_row[m_row] - cam3_row[m_row]).abs()

    #                 print("[CAM INV col] keep", int(m_col.sum()), "max", err_col.max().item(), "mean", err_col.mean().item())
    #                 print("[CAM INV row] keep", int(m_row.sum()), "max", err_row.max().item(), "mean", err_row.mean().item())
            
    #             with torch.no_grad():
    #                 N = xyz_pose.shape[0]
    #                 M = min(N, 20000)
    #                 idx = torch.randint(0, N, (M,), device=device)
    #                 xp = xyz_pose[idx]     # pose points
    #                 xw = xyz_w[idx]        # world points

    #                 Ew = extrinsics[i]     # (4,4)
    #                 K  = intrinsics[i]     # (3,3)

    #                 # Build T (pose->world) in column convention:
    #                 T = torch.eye(4, device=device, dtype=dtype)
    #                 T[:3, :3] = Ri * ki.squeeze()
    #                 T[:3, 3]  = bi.view(-1)

    #                 Epose = Ew @ T

    #                 # --- project_cv outputs ---
    #                 uv1_cv = utils3d.torch.project_cv(xw, Ew, K)[0]     # world + Ew
    #                 uv3_cv = utils3d.torch.project_cv(xp, Epose, K)[0]  # pose + (Ew@T)

    #                 def manual_uv_and_z(x, E, K, variant="col"):
    #                     ones = torch.ones((x.shape[0],1), device=x.device, dtype=x.dtype)
    #                     Xh = torch.cat([x, ones], dim=1)  # [M,4]

    #                     if variant == "col":
    #                         cam = (E @ Xh.t()).t()[:, :3]
    #                     elif variant == "row":
    #                         cam = (Xh @ E.t())[:, :3]
    #                     else:
    #                         raise ValueError

    #                     z = cam[:, 2]

    #                     pix = (K @ cam.t()).t()  # [M,3]
    #                     u = pix[:, 0] / (pix[:, 2] + 1e-8)
    #                     v = pix[:, 1] / (pix[:, 2] + 1e-8)

    #                     uv = torch.stack([u, v], dim=1)
    #                     return uv, z

    #                 # manual projections (try both conventions)
    #                 uv1_col, z1_col = manual_uv_and_z(xw, Ew,    K, "col")
    #                 uv3_col, z3_col = manual_uv_and_z(xp, Epose, K, "col")

    #                 uv1_row, z1_row = manual_uv_and_z(xw, Ew,    K, "row")
    #                 uv3_row, z3_row = manual_uv_and_z(xp, Epose, K, "row")

    #                 # Compare manual vs project_cv (Mode1 + Mode3)
    #                 # Depth-valid mask only (avoid garbage UVs)
    #                 m1 = z1_col > 1e-4
    #                 m3 = z3_col > 1e-4
    #                 m13 = m1 & m3

    #                 err1_col = (uv1_col[m1] - uv1_cv[m1]).abs()
    #                 err3_col = (uv3_col[m3] - uv3_cv[m3]).abs()
    #                 print("[proj check] Mode1 manual(col) vs project_cv: max", err1_col.max().item(), "mean", err1_col.mean().item())
    #                 print("[proj check] Mode3 manual(col) vs project_cv: max", err3_col.max().item(), "mean", err3_col.mean().item())

    #                 m1 = z1_row > 1e-4
    #                 m3 = z3_row > 1e-4
    #                 err1_row = (uv1_row[m1] - uv1_cv[m1]).abs()
    #                 err3_row = (uv3_row[m3] - uv3_cv[m3]).abs()
    #                 print("[proj check] Mode1 manual(row) vs project_cv: max", err1_row.max().item(), "mean", err1_row.mean().item())
    #                 print("[proj check] Mode3 manual(row) vs project_cv: max", err3_row.max().item(), "mean", err3_row.mean().item())

    #                 # Now check UV invariance using the SAME manual projector (no project_cv involved)
    #                 # If math is correct and the manual convention matches the camera, uv1 and uv3 should match.
    #                 m13_col = (z1_col > 1e-4) & (z3_col > 1e-4) & torch.isfinite(uv1_col).all(1) & torch.isfinite(uv3_col).all(1)
    #                 inv_col = (uv1_col[m13_col] - uv3_col[m13_col]).abs()
    #                 print("[UV inv manual(col)] max", inv_col.max().item(), "mean", inv_col.mean().item())

    #                 m13_row = (z1_row > 1e-4) & (z3_row > 1e-4) & torch.isfinite(uv1_row).all(1) & torch.isfinite(uv3_row).all(1)
    #                 inv_row = (uv1_row[m13_row] - uv3_row[m13_row]).abs()
    #                 print("[UV inv manual(row)] max", inv_row.max().item(), "mean", inv_row.mean().item())




    #         rep2.from_xyz(xyz_w)

    #         # (B) rotation (IMPORTANT: use rep.get_rotation, not rep2._rotation)
    #         q_pose = rep.get_rotation if getattr(rep, "_rotation", None) is not None else None
    #         if q_pose is not None and q_pose.shape[-1] == 4:
    #             # Here Ri is the matrix used in xyz_w = x_pose @ Ri^T,
    #             # so Ri is the pose->world rotation in the usual (column) sense.
    #             qR = self._quat_from_rotmat(Ri).view(1, 4).expand_as(q_pose)
    #             q_w = F.normalize(self._quat_mul(qR, q_pose), dim=-1)
    #             rep2.from_rotation(q_w)

    #         # (C) scaling (IMPORTANT: use rep.get_scaling, not rep2._scaling)
    #         if getattr(rep, "_scaling", None) is not None:
    #             s_pose = rep.get_scaling.view(-1, 3)           # (N,3) physical
    #             s_w = s_pose * ki.abs().clamp_min(eps)         # similarity scale
    #             min_k = float(getattr(rep2, "mininum_kernel_size", 0.0))
    #             if min_k > 0:
    #                 s_w = torch.clamp(s_w, min=min_k + 1e-9)
    #             rep2.from_scaling(s_w)

    #         out.append(rep2)

    #     return out


    # @torch.no_grad()
    # def _transform_reps_for_loss(
    #     self,
    #     reps,
    #     *,
    #     R_row, s_aug, t_aug, c0, s_norm, c_norm,
    #     origin=None, voxel_size=None,
    #     intrinsics=None, extrinsics=None,
    #     debug_xyz=True,
    #     max_check_points=20000,
    # ):
    #     """
    #     reps: list length B, each rep in POSE space (unit2).
    #     Returns: list length B, each rep in WORLD/RAW space consistent with extrinsics.

    #     This function applies the SAME affine that debug_camera_frame uses:
    #         xyz_world = (xyz_pose @ R_i^T) * k_i + b_i
    #     where (R_i, k_i, b_i) are produced by self._pose_to_world_params(...).
    #     """

    #     # ---- basic sanity ----
    #     assert isinstance(reps, (list, tuple)), "reps must be a list/tuple of per-sample reps"
    #     B = len(reps)
    #     assert R_row.shape[0] == B, f"R_row batch {R_row.shape[0]} != reps {B}"
    #     for name, x in [("s_aug", s_aug), ("t_aug", t_aug), ("c0", c0), ("s_norm", s_norm), ("c_norm", c_norm)]:
    #         assert x.shape[0] == B, f"{name} batch {x.shape[0]} != reps {B}"

    #     device = reps[0].get_xyz.device
    #     dtype  = reps[0].get_xyz.dtype

    #     # ---- compute per-sample pose->world params (THIS must match your debug code) ----
    #     R, k, b = self._pose_to_world_params(
    #         R_row.to(device=device, dtype=dtype),
    #         s_aug.to(device=device, dtype=dtype),
    #         t_aug.to(device=device, dtype=dtype),
    #         c0.to(device=device, dtype=dtype),
    #         s_norm.to(device=device, dtype=dtype),
    #         c_norm.to(device=device, dtype=dtype),
    #         # If your _pose_to_world_params ALSO needs origin/voxel_size, pass them there instead.
    #     )
    #     # expected shapes:
    #     #   R: (B,3,3)
    #     #   k: (B,) or (B,1) or (B,1,1)
    #     #   b: (B,3)

    #     # normalize k shape to (B,1)
    #     if k.dim() == 1:
    #         k = k.view(B, 1)
    #     elif k.dim() == 2 and k.shape[1] == 1:
    #         pass
    #     elif k.dim() == 3:
    #         k = k.view(B, 1)
    #     else:
    #         raise ValueError(f"Unexpected k shape: {k.shape}")

    #     # normalize b to (B,1,3) for broadcasting
    #     if b.dim() == 2:
    #         b_ = b.view(B, 1, 3)
    #     elif b.dim() == 3 and b.shape[1] == 1:
    #         b_ = b
    #     else:
    #         raise ValueError(f"Unexpected b shape: {b.shape}")

    #     reps_world = []

    #     # ---- apply per sample (NO batch broadcasting) ----
    #     for i in range(B):
    #         rep = reps[i]

    #         xyz = rep.get_xyz.view(-1, 3)  # (N,3) pose
    #         Ri  = R[i]                      # (3,3)
    #         ki  = k[i]                      # (1,)
    #         bi  = b_[i]                     # (1,3)

    #         xyz_w = (xyz @ Ri.t()) * ki + bi  # (N,3)

    #         rep_w = copy.deepcopy(rep)

    #         # Most gaussian reps store xyz in rep._xyz (Parameter). If yours differs, adjust here.
    #         if hasattr(rep_w, "_xyz"):
    #             rep_w._xyz.data = xyz_w.view_as(rep_w._xyz).contiguous()
    #         else:
    #             # fallback: try common name
    #             try:
    #                 rep_w.xyz.data = xyz_w.view_as(rep_w.xyz).contiguous()
    #             except Exception as e:
    #                 raise RuntimeError("Could not set xyz on rep. Please adapt to your rep class.") from e

    #         # Optional: scale gaussians by the same similarity scale
    #         # (comment out if your renderer expects scaling to stay in pose space)
    #         if hasattr(rep_w, "get_scaling") and hasattr(rep_w, "_scaling"):
    #             s = rep.get_scaling
    #             rep_w._scaling.data = (s * ki).view_as(rep_w._scaling).contiguous()

    #         reps_world.append(rep_w)

    #         # ---- built-in XYZ consistency check (the one you are doing manually) ----
    #         if debug_xyz:
    #             # sample points to keep it fast
    #             xyz_w_out = reps_world[i].get_xyz.view(-1, 3)
    #             if xyz_w_out.shape[0] > max_check_points:
    #                 idx = torch.randint(0, xyz_w_out.shape[0], (max_check_points,), device=xyz_w_out.device)
    #                 err = (xyz_w_out[idx] - xyz_w[idx]).abs()
    #             else:
    #                 err = (xyz_w_out - xyz_w).abs()

    #             max_err = float(err.max().item())
    #             mean_err = float(err.mean().item())
    #             print(f"[XYZ CONSIST inside _transform i={i}] max={max_err:.6e} mean={mean_err:.6e}")

    #             # If this is not ~1e-7 for ALL i, you are not actually applying what you think.
    #             # In that case, it’s a rep setter / shape / view issue.
    #             if max_err > 1e-4:
    #                 print(f"⚠️  Large XYZ CONSIST for i={i}. "
    #                     f"Likely: wrong tensor reshaping when writing rep._xyz, or rep.get_xyz != rep._xyz.")

    #     return reps_world

    # @torch.no_grad()
    # def _transform_reps_for_loss(
    #     self,
    #     reps,                       # list[Gaussian] length B (decoder outputs)
    #     *,
    #     R_row, s_aug, t_aug, c0,
    #     s_norm, c_norm,
    #     origin=None, voxel_size=None,   # unused here (kept for signature compatibility)
    #     intrinsics=None, extrinsics=None,  # unused here (kept for signature compatibility)
    #     posed_mask = None,
    #     clamp_xyz01: bool = True,
    #     debug_xyz_consist: bool = False,
    # ):
    #     """
    #     Transforms each rep from POSE space -> WORLD space, *correctly* accounting for
    #     Gaussian.get_xyz being an affine of _xyz via rep.aabb.

    #     Key detail:
    #         rep.get_xyz = rep._xyz * rep.aabb[3:] + rep.aabb[:3]
    #     So to make get_xyz equal to some xyz_target, we must set:
    #         rep._xyz = (xyz_target - aabb_min) / aabb_size

    #     Returns:
    #         reps_world: list[Gaussian] (deep-copied) with updated _xyz
    #     """

    #     B = len(reps)
    #     assert R_row.shape[0] == B, f"R_row batch {R_row.shape[0]} != {B}"
    #     assert s_aug.shape[0] == B
    #     assert t_aug.shape[0] == B
    #     assert c0.shape[0] == B
    #     assert s_norm.shape[0] == B
    #     assert c_norm.shape[0] == B

    #     # Compute pose->world parameters for the whole batch.
    #     # This is your existing helper; it should output:
    #     #   xyz_world = (xyz_pose @ R^T) * k + b
    #     R, k, b = self._pose_to_world_params(
    #         R_row, s_aug, t_aug, c0, s_norm, c_norm
    #     )  # R: (B,3,3), k: (B,1) or (B,), b: (B,3)

    #     reps_world = []

    #     for i in range(B):
    #         rep_src = reps[i]

    #         # Deep copy the Gaussian so we don't modify the original
    #         rep_w = copy.deepcopy(rep_src)

    #         # --- 1) read xyz in POSE space in *world units* (i.e., via get_xyz) ---
    #         xyz_pose = rep_src.get_xyz.view(-1, 3)  # IMPORTANT: not _xyz

    #         # --- 2) apply pose->world ---
    #         Ri = R[i]                       # (3,3)
    #         ki = k[i].view(1, 1)            # (1,1)
    #         bi = b[i].view(1, 3)            # (1,3)
    #         xyz_world_target = (xyz_pose @ Ri.t()) * ki + bi  # (N,3)

    #         # --- 3) write into rep_w._xyz so that rep_w.get_xyz == xyz_world_target ---
    #         # Because get_xyz = _xyz * aabb_size + aabb_min
    #         aabb = rep_w.aabb.to(xyz_world_target.device, xyz_world_target.dtype)
    #         aabb_min  = aabb[:3].view(1, 3)
    #         aabb_size = aabb[3:].view(1, 3).clamp_min(1e-12)

    #         xyz01 = (xyz_world_target - aabb_min) / aabb_size
    #         if clamp_xyz01:
    #             xyz01 = xyz01.clamp(0.0, 1.0)

    #         # Ensure exact same shape + dtype as storage
    #         if not hasattr(rep_w, "_xyz"):
    #             raise AttributeError("Gaussian has no attribute _xyz")
    #         rep_w._xyz = rep_w._xyz.clone()  # avoid weird shared storage
    #         rep_w._xyz.view(-1, 3).copy_(xyz01.to(rep_w._xyz.dtype))

    #         # --- 4) optional sanity check (this is the one that must become ~1e-7) ---
    #         if debug_xyz_consist:
    #             xyz_back = rep_w.get_xyz.view(-1, 3)
    #             err = (xyz_back - xyz_world_target).abs()
    #             mx = float(err.max().item())
    #             me = float(err.mean().item())
    #             print(f"[XYZ CONSIST inside _transform i={i}] max={mx:.6e} mean={me:.6e}")
    #             if mx > 1e-3:
    #                 print("⚠️  Large XYZ CONSIST. This usually means:")
    #                 print("   - you wrote rep._xyz in world units (wrong), OR")
    #                 print("   - rep.aabb is not what you think, OR")
    #                 print("   - rep.get_xyz applies something else besides aabb.")

    #         reps_world.append(rep_w)

    #     return reps_world



    
    

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
        R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)
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
                xyz01 = (xyz_world + 1.0) * 0.5
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

    # def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm):
        # """
        # Returns per-sample (R, k, b) so that:
        # x_w = (x_pose @ R.T) * k + b
        # Shapes:
        # R: (B,3,3)
        # k: (B,1,1)
        # b: (B,1,3)
        # """
        # # ensure float
        # R = R_row
        # k = 1.0 / (s_aug * s_norm)                  # (B,)
        # k = k.view(-1, 1, 1)
        # b = c_norm + c0 / s_norm.unsqueeze(-1) - (t_aug @ R.transpose(1, 2)) * k.squeeze(-1)  # (B,3)
        # b = b.view(-1, 1, 3)
        # return R, k, b
    @torch.no_grad()
    def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm, eps=1e-12):
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

        R = R_row
        return R, k, b


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


    # @torch.no_grad()
    # def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
    #     xyz = torch.cat([g.get_xyz for g in reps], dim=0)
    #     xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['decoder'].resolution - 0.5
    #     offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['decoder'].rep_config['num_gaussians'], -1).reshape(-1, 3)
    #     status = {
    #         'xyz': xyz,
    #         'offset': offset,
    #         'scale': torch.cat([g.get_scaling for g in reps], dim=0),
    #         'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
    #     }

    #     for k in list(status.keys()):
    #         status[k] = {
    #             'mean': status[k].mean().item(),
    #             'max': status[k].max().item(),
    #             'min': status[k].min().item(),
    #         }
            
    #     return status
    def apply_T_col(self, xyz, T):
        # xyz: [N,3] row tensor, but T is column convention
        ones = torch.ones((xyz.shape[0],1), device=xyz.device, dtype=xyz.dtype)
        Xh = torch.cat([xyz, ones], dim=1)                 # [N,4]
        Yh = (T @ Xh.t()).t()                              # [N,4]
        return Yh[:, :3]
    
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


    @torch.no_grad()
    def debug_coords_equivalence(self,feats, reps, origin, voxel_size, res, num_gaussians, i=0):
        """
        feats: SparseTensor input to encoder (same one you pass to decoder)
        reps:  list[Gaussian] output of decoder (before any transform)
        origin: (B,3)
        voxel_size: (B,) or (B,1)
        res: int (64)
        num_gaussians: e.g. 32
        """
        device = reps[i]._xyz.device
        origin_i = origin[i].to(device).float().view(1, 3)
        vs_i = voxel_size[i].to(device).float().view(1, 1)
        rep = reps[i]

        # -------------------------
        # (1) SparseTensor coords (what decoder used)
        # -------------------------
        # These are the integer grid indices per occupied voxel.
        coords = feats.coords[feats.layout[i]][:, 1:]  # (Nvox,3) int-ish
        coords_f = coords.to(device).float()

        print("\n[DEBUG] sample", i)
        print("coords shape:", tuple(coords.shape))
        print("coords min:", coords.min(0).values.tolist(), "max:", coords.max(0).values.tolist())

        # convert coords -> xyz01 according to the decoder rule
        xyz01_from_coords = (coords_f + 0.5) / float(res)  # (Nvox,3) in [0,1] ideally

        # -------------------------
        # (2) rep._xyz voxel centers (average over gaussians)
        # -------------------------
        # rep._xyz is (Nvox*num_gaussians,3) in your decoder.
        xyz01_rep = rep._xyz.view(-1, 3)
        if xyz01_rep.shape[0] % num_gaussians != 0:
            print("!! rep._xyz length not divisible by num_gaussians:",
                xyz01_rep.shape[0], num_gaussians)
            return

        Nrep_vox = xyz01_rep.shape[0] // num_gaussians
        xyz01_rep_vox = xyz01_rep.view(Nrep_vox, num_gaussians, 3).mean(1)  # (Nrep_vox,3)

        print("rep._xyz vox count:", Nrep_vox)

        # If things are consistent, Nrep_vox should equal coords.shape[0]
        if Nrep_vox != coords.shape[0]:
            print("!! MISMATCH: Nvox from coords =", coords.shape[0], "but rep implies", Nrep_vox)
            print("   This usually means coords got rebatched/rebased OR you changed voxel selection.")
        else:
            d = (xyz01_rep_vox - xyz01_from_coords).abs()
            print("xyz01(rep) vs xyz01(coords): max", float(d.max()), "mean", float(d.mean()))
            # If reindexing happened, you'll see a *constant offset* in xyz01:
            delta = (xyz01_rep_vox - xyz01_from_coords).median(0).values
            print("median delta xyz01(rep - coords):", delta.tolist())

        # -------------------------
        # (3) Check posed-unit2 equivalence:
        #     rep.get_xyz  vs origin + rep._xyz*(voxel_size*res)
        # -------------------------
        xyz_pose_from_origin = origin_i + xyz01_rep_vox * (vs_i * float(res))   # (Nvox,3)
        xyz_pose_from_aabb   = rep.get_xyz.view(-1, 3).view(Nrep_vox, num_gaussians, 3).mean(1)

        dd = (xyz_pose_from_origin - xyz_pose_from_aabb).abs()
        print("pose(unit2) origin-map vs rep.get_xyz: max", float(dd.max()), "mean", float(dd.mean()))

        # Helpful sanity prints:
        print("origin:", origin_i.squeeze(0).tolist(), "voxel_size:", float(vs_i.squeeze()), "vs*res:", float((vs_i*res).squeeze()))
        print("rep.aabb:", rep.aabb.detach().cpu().tolist())
        print("rep.get_xyz min/max:", rep.get_xyz.min(0).values.tolist(), rep.get_xyz.max(0).values.tolist())
        print("origin-map pose min/max:", xyz_pose_from_origin.min(0).values.tolist(), xyz_pose_from_origin.max(0).values.tolist())

    @torch.no_grad()
    def debug_camera_frame(
        self,
        rep, alpha_img, E_world, K,
        R_row, s_aug, t_aug, c0, s_norm, c_norm,
        max_points=20000
    ):
        import torch
        import utils3d

        device = rep._xyz.device
        E_world = E_world.to(device)
        K = K.to(device)

        # ---- points in POSE space (unit2) ----
        xyz_pose = rep.get_xyz.view(-1, 3)  # (N,3)

        if xyz_pose.shape[0] > max_points:
            idx = torch.randint(0, xyz_pose.shape[0], (max_points,), device=device)
            xyz_pose = xyz_pose[idx]

        # ---- build pose->world affine: x_w = (x_pose @ R.T) * k + b ----
        R, k, b = self._pose_to_world_params(
            R_row.view(1,3,3),
            s_aug.view(1),
            t_aug.view(1,3),
            c0.view(1,3),
            s_norm.view(1),
            c_norm.view(1,3),
        )
        R = R[0]
        k = k[0].view(1,1)          # scalar
        b = b[0].view(1,3)

        xyz_world = (xyz_pose @ R.t()) * k + b  # (N,3)

        # ---- helpers ----
        def project_uv01(xyz, E):
            # force fp32 for stable comparisons
            uv01 = utils3d.torch.project_cv(xyz.float(), E.float(), K.float())[0]
            return uv01

        def cam_xyz(xyz, E):
            ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
            Xh = torch.cat([xyz, ones], dim=1)           # (N,4)
            C = (E @ Xh.t()).t()[:, :3]                  # (N,3)
            return C

        def alpha_coverage(uv01, alpha, thr=0.05):
            if alpha.dtype not in (torch.float16, torch.float32):
                alpha = alpha.float()
            if alpha.max() > 1.5:
                alpha = alpha / 255.0
            H, W = alpha.shape[-2], alpha.shape[-1]

            m = torch.isfinite(uv01).all(dim=1)
            uv = uv01[m]
            if uv.numel() == 0:
                return 0.0

            xs = torch.clamp((uv[:,0] * (W-1)).round().long(), 0, W-1)
            ys = torch.clamp((uv[:,1] * (H-1)).round().long(), 0, H-1)
            inside = alpha[ys, xs] > thr
            return float(inside.float().mean().item())

        # ---- Mode 1 ----
        uv_w = project_uv01(xyz_world, E_world)
        cov_w = alpha_coverage(uv_w, alpha_img)

        # ---- Mode 2 ----
        uv_p_direct = project_uv01(xyz_pose, E_world)
        cov_p_direct = alpha_coverage(uv_p_direct, alpha_img)

        # ---- Mode 3 ----
        T = torch.eye(4, device=device, dtype=E_world.dtype)
        # row convention x_w = x_pose @ (k*R^T) + b  => column A = k*R
        # BUT since your xyz_world uses (xyz_pose @ R.t()) * k + b,
        # the column-form matrix is: A = (k * R)
        # and T[:3,3] = b
        T[:3,:3] = (k.squeeze() * R)        # <-- this is the important correction
        T[:3, 3] = b.squeeze(0)
        E_pose = E_world @ T

        uv_p_comp = project_uv01(xyz_pose, E_pose)
        cov_p_comp = alpha_coverage(uv_p_comp, alpha_img)

        # ---- UV invariance check: add depth validity ----
        cam1 = cam_xyz(xyz_world.float(), E_world.float())
        cam3 = cam_xyz(xyz_pose.float(), E_pose.float())
        z1 = cam1[:, 2]
        z3 = cam3[:, 2]

        m = (
            torch.isfinite(uv_w).all(dim=1) &
            torch.isfinite(uv_p_comp).all(dim=1) &
            (z1 > 1e-4) & (z3 > 1e-4)
        )

        if m.any():
            uv_err = (uv_w[m] - uv_p_comp[m]).abs()
            max_err = float(uv_err.max().item())
            mean_err = float(uv_err.mean().item())
            keep = int(m.sum().item())
        else:
            max_err, mean_err, keep = float("nan"), float("nan"), 0

        print("alpha coverage:")
        print("  Mode1  xyz_world + E_world      :", cov_w)
        print("  Mode2  xyz_pose  + E_world      :", cov_p_direct)
        print("  Mode3  xyz_pose  + (E_world@T)  :", cov_p_comp)
        print(f"UV invariance (Mode1 vs Mode3, masked z>0): max={max_err:.3e} mean={mean_err:.3e} keep={keep}/{xyz_pose.shape[0]}")

        return {
            "cov_world": cov_w,
            "cov_pose_direct": cov_p_direct,
            "cov_pose_comp": cov_p_comp,
            "uv_err_max": max_err,
            "uv_err_mean": mean_err,
            "uv_keep": keep,
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

    
    # @torch.no_grad()
    # def run_snapshot(
    #     self,
    #     num_samples: int,
    #     batch_size: int,
    #     verbose: bool = False,
    # ) -> Dict:
    #     dataloader = DataLoader(
    #         copy.deepcopy(self.dataset),
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=0,
    #         collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
    #     )
    #     num_samples = 4
    #     # inference
    #     ret_dict = {}
    #     gt_images = []
    #     exts = []
    #     ints = []
    #     pose_pack = { "R_row": [], "s_aug": [], "t_aug": [], "c0": [], "s_norm": [], "c_norm": [] , "origin": [], "voxel_size": []}
    #     reps = []
    #     alphas = []

    #     for i in range(0, num_samples, batch_size):
    #         batch = min(batch_size, num_samples - i)
    #         data = next(iter(dataloader))
    #         args = {k: v[:batch].cuda() for k, v in data.items()}

    #         gt_images.append(args['image'] * args['alpha'][:, None])
    #         exts.append(args['extrinsics'])
    #         ints.append(args['intrinsics'])
    #         alphas.append(args['alpha'])

    #         for k in pose_pack:
    #             pose_pack[k].append(args[k])

    #         z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
    #         reps.extend(self.models['decoder'](z))

    #     gt_images = torch.cat(gt_images, dim=0)
    #     alphas = torch.cat(alphas, dim=0)
    #     ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
    #     exts = torch.cat(exts, dim=0)
    #     ints = torch.cat(ints, dim=0)
    #     for k in pose_pack:
    #         pose_pack[k] = torch.cat(pose_pack[k], dim=0)
    #     reps_world = self._transform_reps_for_loss(
    #         reps,
    #         R_row=pose_pack["R_row"],
    #         s_aug=pose_pack["s_aug"],
    #         t_aug=pose_pack["t_aug"],
    #         c0=pose_pack["c0"],
    #         s_norm=pose_pack["s_norm"],
    #         c_norm=pose_pack["c_norm"],
    #         origin=pose_pack["origin"],
    #         voxel_size=pose_pack["voxel_size"],
    #         intrinsics=ints,
    #         extrinsics=exts,
    #     )

    #     # ---- DEBUG: verify camera frame / mask coverage on sample 0 ----
    #     for j in range(num_samples):
    #         alpha_img = (args["alpha"][j] if "alpha" in args else None)  # careful: args is last batch only
    #         # Better: rebuild alpha from gt_images you already collected:
    #         alpha_img = (gt_images[j].sum(0) > 0).float()  # rough if gt_images is masked RGB, not true alpha

    #         # If you have the real alpha tensor separately, store it like you did with gt_images:
    #         # e.g. alpha_list.append(args["alpha"]) and then alpha = torch.cat(alpha_list)

    #         self.debug_camera_frame(
    #             rep=reps[j],                       # IMPORTANT: this rep is POSE space
    #             alpha_img=args["alpha"][j],        # use the true alpha from dataset
    #             E_world=exts[j],
    #             K=ints[j],
    #             R_row=pose_pack["R_row"][j],
    #             s_aug=pose_pack["s_aug"][j],
    #             t_aug=pose_pack["t_aug"][j],
    #             c0=pose_pack["c0"][j],
    #             s_norm=pose_pack["s_norm"][j],
    #             c_norm=pose_pack["c_norm"][j],
    #         )
    #         with torch.no_grad():
    #             H, W = alphas[j].shape[-2], alphas[j].shape[-1]
    #             ys, xs = torch.where(alphas[j] > 0.1)
    #             if xs.numel() > 0:
    #                 alpha_c = torch.stack([xs.float().mean(), ys.float().mean()])  # (2,) in px
    #             else:
    #                 alpha_c = torch.tensor([float("nan"), float("nan")], device=alphas[j].device)

    #             # project world points (Mode1)
    #             xyz_pose = reps[j].get_xyz.view(-1,3)
    #             R, k, b = self._pose_to_world_params(
    #                 pose_pack["R_row"][j:j+1],
    #                 pose_pack["s_aug"][j:j+1],
    #                 pose_pack["t_aug"][j:j+1],
    #                 pose_pack["c0"][j:j+1],
    #                 pose_pack["s_norm"][j:j+1],
    #                 pose_pack["c_norm"][j:j+1],
    #             )
    #             xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0]

    #             uv01 = utils3d.torch.project_cv(xyz_world, exts[j], ints[j])[0]  # (N,2)
    #             m = torch.isfinite(uv01).all(1)
    #             uv01 = uv01[m]
    #             uv_px = uv01 * torch.tensor([W-1, H-1], device=uv01.device, dtype=uv01.dtype)
    #             proj_c = uv_px.mean(0)

    #             print("alpha centroid px:", alpha_c.tolist())
    #             print("proj centroid px :", proj_c.tolist())
    #             print("centroid delta px:", (proj_c - alpha_c).tolist())

    #             H, W = alphas[j].shape[-2], alphas[j].shape[-1]

    #             # alpha centroid
    #             ys, xs = torch.where(alphas[j] > 0.1)
    #             alpha_c = torch.stack([xs.float().mean(), ys.float().mean()]) if xs.numel() else torch.tensor([float("nan"), float("nan")], device=uv01.device)

    #             # world points
    #             xyz_pose = reps[j].get_xyz.view(-1,3)
    #             R, k, b = self._pose_to_world_params(
    #                 pose_pack["R_row"][j:j+1],
    #                 pose_pack["s_aug"][j:j+1],
    #                 pose_pack["t_aug"][j:j+1],
    #                 pose_pack["c0"][j:j+1],
    #                 pose_pack["s_norm"][j:j+1],
    #                 pose_pack["c_norm"][j:j+1],
    #             )
    #             xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0]

    #             # project
    #             uv01 = utils3d.torch.project_cv(xyz_world, exts[j], ints[j])[0]  # (N,2)
    #             m = torch.isfinite(uv01).all(1)

    #             # weights: opacity (optionally multiply by volume proxy)
    #             w = reps_world[j].get_opacity.view(-1)  # (N,)
    #             # Optional: emphasize big gaussians too:
    #             # s = reps_world[j].get_scaling  # (N,3)
    #             # vol = (s[:,0] * s[:,1] * s[:,2]).clamp_min(1e-12)
    #             # w = w * vol.sqrt()

    #             # keep only valid projections + meaningful weights
    #             m = m & (w > 0.01)
    #             uv01 = uv01[m]
    #             w = w[m]

    #             uv_px = uv01 * torch.tensor([W-1, H-1], device=uv01.device, dtype=uv01.dtype)

    #             if uv_px.shape[0] > 0:
    #                 wsum = w.sum().clamp_min(1e-8)
    #                 proj_c_w = (uv_px * w[:, None]).sum(0) / wsum
    #             else:
    #                 proj_c_w = torch.tensor([float("nan"), float("nan")], device=uv01.device)

    #             print("alpha centroid px:", alpha_c.tolist())
    #             print("proj centroid px (weighted):", proj_c_w.tolist())
    #             print("centroid delta px (weighted):", (proj_c_w - alpha_c).tolist())


    #     def centroid_px_from_mask(mask: torch.Tensor):
    #         ys, xs = torch.where(mask)
    #         if xs.numel() == 0:
    #             return torch.tensor([float("nan"), float("nan")], device=mask.device)
    #         return torch.stack([xs.float().mean(), ys.float().mean()], dim=0)

    #     # --- Render twice to estimate alpha ---
    #     self.renderer.rendering_options.resolution = gt_images.shape[-1]

    #     # 1) black bg
    #     self.renderer.rendering_options.bg_color = (0, 0, 0)
    #     rb = self._render_batch(reps_world, exts, ints)
    #     cb = rb["color"]  # (B,3,H,W), assumed in [0,1]

    #     # 2) white bg
    #     self.renderer.rendering_options.bg_color = (1, 1, 1)
    #     rw = self._render_batch(reps_world, exts, ints)
    #     cw = rw["color"]

    #     # Estimate alpha: alpha = 1 - (cw - cb)
    #     # Use mean over channels to reduce noise
    #     alpha_pred = 1.0 - (cw - cb).mean(dim=1)            # (B,H,W)
    #     alpha_pred = alpha_pred.clamp(0.0, 1.0)

    #     # Restore bg setting you want for snapshot display
    #     self.renderer.rendering_options.bg_color = (0, 0, 0)

    #     # --- Centroid test: GT alpha vs Pred alpha ---
    #     B = alpha_pred.shape[0]
    #     H, W = alpha_pred.shape[-2], alpha_pred.shape[-1]
    #     for j in range(min(num_samples, B)):
    #         gt_a = alphas[j]
    #         if gt_a.max() > 1.5:
    #             gt_a = gt_a / 255.0

    #         gt_mask   = gt_a > 0.1
    #         pred_mask = alpha_pred[j] > 0.05

    #         c_gt   = centroid_px_from_mask(gt_mask)
    #         c_pred = centroid_px_from_mask(pred_mask)

    #         print(f"[alpha-centroid j={j}] GT={c_gt.tolist()}  PRED={c_pred.tolist()}  delta(px)={(c_pred-c_gt).tolist()}")

    #         # coverage = fraction of predicted pixels that land inside GT alpha
    #         ys, xs = torch.where(pred_mask)
    #         if xs.numel() > 0:
    #             inside = gt_mask[ys, xs].float().mean().item()
    #         else:
    #             inside = 0.0
    #         print(f"[alpha-coverage j={j}] inside_GT={inside:.4f}")

    #     def weighted_centroid_from_alpha(alpha_map: torch.Tensor, eps=1e-8):
    #         # alpha_map: (H,W) in [0,1]
    #         H, W = alpha_map.shape[-2], alpha_map.shape[-1]
    #         a = alpha_map.clamp_min(0).float()

    #         s = a.sum().clamp_min(eps)
    #         xs = torch.arange(W, device=a.device, dtype=a.dtype)[None, :].expand(H, W)
    #         ys = torch.arange(H, device=a.device, dtype=a.dtype)[:, None].expand(H, W)

    #         cx = (a * xs).sum() / s
    #         cy = (a * ys).sum() / s
    #         return torch.stack([cx, cy], dim=0), s, a.max()

    #     def weighted_inside_gt(alpha_pred: torch.Tensor, gt_mask: torch.Tensor, eps=1e-8):
    #         # "how much predicted alpha lies inside the GT silhouette"
    #         a = alpha_pred.clamp_min(0).float()
    #         num = (a * gt_mask.float()).sum()
    #         den = a.sum().clamp_min(eps)
    #         return (num / den).item()

    #     for j in range(num_samples):
    #         gt_a = alphas[j]
    #         if gt_a.max() > 1.5:
    #             gt_a = gt_a / 255.0
    #         gt_mask = gt_a > 0.1

    #         c_gt, s_gt, mx_gt = weighted_centroid_from_alpha(gt_a)
    #         c_pred, s_pred, mx_pred = weighted_centroid_from_alpha(alpha_pred[j])

    #         inside_w = weighted_inside_gt(alpha_pred[j], gt_mask)

    #         print(f"[alpha-centroid-W j={j}] GT={c_gt.tolist()}  PRED={c_pred.tolist()}  delta(px)={(c_pred-c_gt).tolist()}")
    #         print(f"[alpha-mass j={j}] sum_pred={float(s_pred):.3e} max_pred={float(mx_pred):.3e}  insideGT_weighted={inside_w:.4f}")
    #         H, W = alphas[j].shape[-2], alphas[j].shape[-1]
    #         with torch.no_grad():
    #             xyz_pose = reps[j].get_xyz.view(-1,3)
    #             R, k, b = self._pose_to_world_params(
    #                 pose_pack["R_row"][j:j+1],
    #                 pose_pack["s_aug"][j:j+1],
    #                 pose_pack["t_aug"][j:j+1],
    #                 pose_pack["c0"][j:j+1],
    #                 pose_pack["s_norm"][j:j+1],
    #                 pose_pack["c_norm"][j:j+1],
    #             )
    #             xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0]

    #             uv01 = utils3d.torch.project_cv(xyz_world, exts[j], ints[j])[0]  # (N,2)
    #             m = torch.isfinite(uv01).all(1)

    #             w = reps_world[j].get_opacity.view(-1)
    #             m = m & (w > 0.01)

    #             uv01 = uv01[m]
    #             w = w[m]

    #             uv_px = uv01 * torch.tensor([W-1, H-1], device=uv01.device, dtype=uv01.dtype)
    #             c_proj = (uv_px * w[:,None]).sum(0) / (w.sum().clamp_min(1e-8))

    #             print(f"[proj-centroid-W j={j}] {c_proj.tolist()}")


    #     # self.renderer.rendering_options.bg_color = (0, 0, 0)
    #     # self.renderer.rendering_options.resolution = gt_images.shape[-1]
    #     # def _centroid_px_from_mask(mask: torch.Tensor):
    #     #     # mask: (H,W) bool or 0/1 float
    #     #     ys, xs = torch.where(mask)
    #     #     if xs.numel() == 0:
    #     #         return torch.tensor([float("nan"), float("nan")], device=mask.device)
    #     #     return torch.stack([xs.float().mean(), ys.float().mean()], dim=0)  # (2,) [x,y] in px

    #     # render_results = self._render_batch(reps_world, exts, ints)
    #     # # --- Alpha centroid test: GT alpha vs Pred alpha (rendered) ---
    #     # # You need an alpha/accumulation output from the renderer.
    #     # pred_alpha = None
    #     # for key in ["alpha", "accum", "opacity", "mask"]:
    #     #     if key in render_results:
    #     #         pred_alpha = render_results[key]
    #     #         break

    #     # if pred_alpha is None:
    #     #     print("[alpha-centroid] Renderer did not return alpha/accum. Available keys:", list(render_results.keys()))
    #     # else:
    #     #     # shapes should be (B, H, W) or (B,1,H,W)
    #     #     if pred_alpha.dim() == 4:
    #     #         pred_alpha = pred_alpha[:, 0]  # (B,H,W)

    #     #     B = pred_alpha.shape[0]
    #     #     H, W = pred_alpha.shape[-2], pred_alpha.shape[-1]

    #     #     for j in range(min(num_samples, B)):
    #     #         gt_a = alphas[j]
    #     #         if gt_a.max() > 1.5:  # if 0..255
    #     #             gt_a = gt_a / 255.0
    #     #         gt_mask = gt_a > 0.1

    #     #         pa = pred_alpha[j]
    #     #         if pa.max() > 1.5:
    #     #             pa = pa / 255.0
    #     #         pred_mask = pa > 0.05

    #     #         c_gt   = _centroid_px_from_mask(gt_mask)
    #     #         c_pred = _centroid_px_from_mask(pred_mask)
    #     #         d = c_pred - c_gt

    #     #         print(f"[alpha-centroid j={j}] GT={c_gt.tolist()}  PRED={c_pred.tolist()}  delta(px)={d.tolist()}")


    #     ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})


    #     # render multiview
    #     self.renderer.rendering_options.resolution = 512
    #     ## Build camera
    #     yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    #     yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
    #     yaws = [y + yaws_offset for y in yaws]
    #     pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

    #     ## render each view
    #     miltiview_images = []
    #     for yaw, pitch in zip(yaws, pitch):
    #         orig = torch.tensor([
    #             np.sin(yaw) * np.cos(pitch),
    #             np.cos(yaw) * np.cos(pitch),
    #             np.sin(pitch),
    #         ]).float().cuda() * 2
    #         fov = torch.deg2rad(torch.tensor(30)).cuda()
    #         extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
    #         intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    #         extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
    #         intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
    #         render_results = self._render_batch(reps_world, extrinsics, intrinsics)
    #         miltiview_images.append(render_results['color'])

    #     ## Concatenate views
    #     miltiview_images = torch.cat([
    #         torch.cat(miltiview_images[:2], dim=-2),
    #         torch.cat(miltiview_images[2:], dim=-2),
    #     ], dim=-1)
    #     ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})

    #     self.renderer.rendering_options.bg_color = 'random'
                                    
    #     return ret_dict


    @torch.no_grad()
    def debug_gt_alpha_vs_camera(
        self,
        coords_ijk: torch.Tensor,   # (M,3) int, in [0,res-1]
        alpha_gt: torch.Tensor,     # (H,W) float in [0,1] (or 0..255)
        E_world: torch.Tensor,      # (4,4)
        K: torch.Tensor,            # (3,3)
        R_row, s_aug, t_aug, c0, s_norm, c_norm,
        res: int = 64,
        thr: float = 0.1
    ):
        device = alpha_gt.device
        E_world = E_world.to(device)
        K = K.to(device)

        # alpha centroid
        a = alpha_gt.float()
        if a.max() > 1.5:
            a = a / 255.0
        ys, xs = torch.where(a > thr)
        if xs.numel() == 0:
            print("[GT alpha] empty mask")
            return
        c_alpha = torch.stack([xs.float().mean(), ys.float().mean()], dim=0)

        # voxel -> xyz_pose (unit2) using aabb mapping (-1 + 2*xyz01)
        coords = coords_ijk.to(device).float()
        xyz01 = (coords + 0.5) / float(res)          # center of voxel
        xyz_pose = -1.0 + 2.0 * xyz01                # because aabb=[-1,-1,-1,2,2,2]

        # pose -> world
        R, k, b = self._pose_to_world_params(
            R_row.view(1,3,3), s_aug.view(1), t_aug.view(1,3),
            c0.view(1,3), s_norm.view(1), c_norm.view(1,3)
        )
        xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0]   # (M,3)

        # project
        H, W = a.shape[-2], a.shape[-1]
        uv01 = utils3d.torch.project_cv(xyz_world, E_world, K)[0]  # (M,2)
        m = torch.isfinite(uv01).all(1)
        uv01 = uv01[m]
        if uv01.numel() == 0:
            print("[GT vox] no finite projections")
            return

        uv_px = uv01 * torch.tensor([W-1, H-1], device=device, dtype=uv01.dtype)
        c_proj = uv_px.mean(0)

        d = c_proj - c_alpha
        print(f"[GT alpha centroid] {c_alpha.tolist()}")
        print(f"[GT vox proj centroid] {c_proj.tolist()}")
        print(f"[delta px] {d.tolist()}  (norm {float(d.norm()):.2f}px)")


    @torch.no_grad()
    def debug_render_vs_proj_centroid(
        self,
        rep_world,          # Gaussian in WORLD coords (the one you pass to renderer)
        E_world, K,
        alpha_gt=None,      # optional (H,W)
        res=None,
        max_points=20000,
        thr_img=0.02,       # threshold on rendered intensity
    ):
        import torch
        import torch.nn.functional as F

        device = rep_world.get_xyz.device
        E_world = E_world.to(device)
        K = K.to(device)

        # ---------- 1) projection centroid from the *same* world gaussians ----------
        xyz_w = rep_world.get_xyz.view(-1, 3)
        opa   = rep_world.get_opacity.view(-1)

        # subsample for speed (keep higher opacity more often)
        N = xyz_w.shape[0]
        if N > max_points:
            # importance sampling by opacity
            p = opa.clamp_min(0)
            p = p / (p.sum().clamp_min(1e-8))
            idx = torch.multinomial(p, max_points, replacement=False)
            xyz_w = xyz_w[idx]
            opa   = opa[idx]

        uv01 = utils3d.torch.project_cv(xyz_w, E_world, K)[0]  # (M,2)
        m = torch.isfinite(uv01).all(1) & (opa > 0.01)
        uv01 = uv01[m]
        w    = opa[m]

        # If nothing valid, bail
        if uv01.numel() == 0:
            print("[proj-centroid] no valid points")
            return

        # Need H,W to convert to pixels (take from alpha_gt if available; else from renderer resolution)
        if alpha_gt is not None:
            H, W = alpha_gt.shape[-2], alpha_gt.shape[-1]
        else:
            assert res is not None, "pass alpha_gt or res"
            H = W = int(res)

        uv_px = uv01 * torch.tensor([W - 1, H - 1], device=device, dtype=uv01.dtype)
        c_proj = (uv_px * w[:, None]).sum(0) / (w.sum().clamp_min(1e-8))

        # ---------- 2) render on black bg and compute centroid from image energy ----------
        old_bg = self.renderer.rendering_options.bg_color
        old_res = self.renderer.rendering_options.resolution

        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = H

        rb = self._render_batch([rep_world], E_world.unsqueeze(0), K.unsqueeze(0))
        img = rb["color"][0]  # (3,H,W)

        # intensity mask (robust silhouette proxy)
        inten = img.mean(0)   # (H,W) in [0,1] approx
        mask = inten > thr_img

        ys, xs = torch.where(mask)
        if xs.numel() == 0:
            c_img = torch.tensor([float("nan"), float("nan")], device=device)
        else:
            # weighted centroid by intensity
            ww = inten[ys, xs].clamp_min(1e-8)
            cx = (xs.float() * ww).sum() / ww.sum()
            cy = (ys.float() * ww).sum() / ww.sum()
            c_img = torch.stack([cx, cy], dim=0)

        # restore renderer settings
        self.renderer.rendering_options.bg_color = old_bg
        self.renderer.rendering_options.resolution = old_res

        d = (c_img - c_proj)
        print(f"[render-centroid img] {c_img.tolist()}")
        print(f"[proj-centroid xyz]  {c_proj.tolist()}")
        print(f"[delta px]           {d.tolist()}  (norm {float(d.norm()):.2f}px)")

        # ---------- 3) optional: compare to GT alpha centroid ----------
        if alpha_gt is not None:
            a = alpha_gt.float()
            if a.max() > 1.5:
                a = a / 255.0
            ys2, xs2 = torch.where(a > 0.1)
            if xs2.numel() > 0:
                c_gt = torch.stack([xs2.float().mean(), ys2.float().mean()], dim=0)
                print(f"[gt-alpha centroid] {c_gt.tolist()}")
                print(f"[img - gt]          {(c_img - c_gt).tolist()} (norm {float((c_img-c_gt).norm()):.2f}px)")
                print(f"[proj - gt]         {(c_proj - c_gt).tolist()} (norm {float((c_proj-c_gt).norm()):.2f}px)")

    
    @torch.no_grad()
    def nn_err(self, a_xyz, b_xyz, max_points=20000):
        a = a_xyz.detach().cpu().numpy()
        b = b_xyz.detach().cpu().numpy()

        if a.shape[0] > max_points:
            a = a[np.random.choice(a.shape[0], max_points, replace=False)]
        if b.shape[0] > max_points:
            b = b[np.random.choice(b.shape[0], max_points, replace=False)]
        tree = cKDTree(b)
        d, _ = tree.query(a, k=1)
        return float(d.max()), float(d.mean())
    
    @torch.no_grad()
    def nn_err_query_subset(self, a_xyz, b_xyz, max_query=20000):
        """
        Query points from a (possibly subsampled) against ALL points of b.
        This is permutation-invariant and meaningful.
        """
        a = a_xyz.detach().cpu().numpy()
        b = b_xyz.detach().cpu().numpy()

        if a.shape[0] > max_query:
            idx = np.random.choice(a.shape[0], size=max_query, replace=False)
            a = a[idx]

        tree = cKDTree(b)  # build on full b
        d, _ = tree.query(a, k=1)
        return float(d.max()), float(d.mean())
    
    @torch.no_grad()
    def diagnose_which_transform(self, rep_pose, rep_world, j, pose_pack):
        device = rep_pose.get_xyz.device
        dtype  = rep_pose.get_xyz.dtype

        xyz_pose = rep_pose.get_xyz.view(-1, 3)
        xyz_w    = rep_world.get_xyz.view(-1, 3)

        # pull params
        R_row = pose_pack["R_row"][j].to(device=device, dtype=dtype)
        s_aug = pose_pack["s_aug"][j].to(device=device, dtype=dtype)
        t_aug = pose_pack["t_aug"][j].to(device=device, dtype=dtype)
        c0    = pose_pack["c0"][j].to(device=device, dtype=dtype)
        s_norm= pose_pack["s_norm"][j].to(device=device, dtype=dtype)
        c_norm= pose_pack["c_norm"][j].to(device=device, dtype=dtype)

        # Candidate A: exact pipeline pose->world (should win)
        R, k, b = self._pose_to_world_params(
            pose_pack["R_row"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["s_aug"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["t_aug"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["c0"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["s_norm"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["c_norm"][j:j+1].to(device=device, dtype=dtype),
        )
        xyz_A = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0].view(1, 3)

        # Candidate B: pose aug only (ignoring norm)
        # This is: x_norm = ((x_pose - t)/s) * R + c0   (column version of inverse)
        Ainv = (1.0 / s_aug) * R_row
        binv = c0 - Ainv @ t_aug
        xyz_B = (xyz_pose @ Ainv.t()) + binv.view(1, 3)  # careful: row points => @ Ainv^T

        # Candidate C: identity
        xyz_C = xyz_pose

        Amax, Amean = self.nn_err_query_subset(xyz_A, xyz_w)
        Bmax, Bmean = self.nn_err_query_subset(xyz_B, xyz_w)
        Cmax, Cmean = self.nn_err_query_subset(xyz_C, xyz_w)

        print(f"[which j={j}] NN(A pipeline pose->world): max={Amax:.3e} mean={Amean:.3e}")
        print(f"[which j={j}] NN(B pose->norm only)      : max={Bmax:.3e} mean={Bmean:.3e}")
        print(f"[which j={j}] NN(C identity)             : max={Cmax:.3e} mean={Cmean:.3e}")



    @torch.no_grad()
    def debug_transform_consistency_one(
        self, rep_pose, rep_world, j,
        pose_pack,
    ):
        device = rep_pose.get_xyz.device
        dtype  = rep_pose.get_xyz.dtype

        # points in POSE space (already aabb-mapped via get_xyz)
        xyz_pose = rep_pose.get_xyz.view(-1, 3)

        # Use the *same* pose->world params your pipeline uses
        # This must match _transform_reps_for_loss exactly.
        R, k, b = self._pose_to_world_params(
            pose_pack["R_row"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["s_aug"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["t_aug"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["c0"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["s_norm"][j:j+1].to(device=device, dtype=dtype),
            pose_pack["c_norm"][j:j+1].to(device=device, dtype=dtype),
        )
        # R: [1,3,3], k: [1,1], b: [1,3] (based on your prints)

        xyz_world_ref = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0].view(1, 3)

        # points produced by _transform_reps_for_loss
        xyz_world_loss = rep_world.get_xyz.view(-1, 3)

        # If ordering is not guaranteed, don't do per-index max/mean.
        # Use NN only (order-invariant).
        max_nn, mean_nn = self.nn_err_query_subset(xyz_world_ref, xyz_world_loss)
        # direct check (same point order)
        n = min(xyz_world_ref.shape[0], xyz_world_loss.shape[0])
        direct = (xyz_world_ref[:n] - xyz_world_loss[:n]).abs()
        print(f"[XYZ DIRECT j={j}] max={direct.max().item():.6e} mean={direct.mean().item():.6e}")

        print(f"[XYZ CONSIST j={j}] NN max={max_nn:.6e} mean={mean_nn:.6e}  "
            f"N_pose={xyz_pose.shape[0]} N_world={xyz_world_loss.shape[0]}")

        return max_nn, mean_nn

    @torch.no_grad()
    def inspect_rep_xyz(self, rep, name="rep"):
        x = rep.get_xyz
        print(f"[{name}] get_xyz shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"min={float(x.min()):.4f} max={float(x.max()):.4f} ptr={x.data_ptr()}")

        # common candidate attributes
        for k in ["_xyz", "xyz", "means3D", "means", "mu", "pos"]:
            if hasattr(rep, k):
                v = getattr(rep, k)
                if torch.is_tensor(v):
                    print(f"  attr {k:8s} shape={tuple(v.shape)} ptr={v.data_ptr()} "
                        f"maxdiff(get_xyz, {k})={float((x.reshape(-1,3) - v.reshape(-1,3)).abs().max()):.6e}")
                else:
                    print(f"  attr {k:8s} type={type(v)}")

        # also list any tensor attrs containing 'xyz'
        hits = []
        for k, v in rep.__dict__.items():
            if "xyz" in k.lower() and torch.is_tensor(v):
                hits.append(k)
        if hits:
            print("  tensor attrs containing 'xyz':", hits)

    # @torch.no_grad()
    # def run_snapshot(
    #     self,
    #     num_samples: int,
    #     batch_size: int,
    #     verbose: bool = False,
    # ) -> Dict:
    #     import copy
    #     import numpy as np
    #     import torch
    #     from torch.utils.data import DataLoader

    #     # -------------------------
    #     # helpers
    #     # -------------------------
    #     def weighted_centroid_from_alpha(alpha_map: torch.Tensor, eps=1e-8):
    #         # alpha_map: (H,W) float in [0,1]
    #         a = alpha_map.float().clamp_min(0.0)
    #         H, W = a.shape[-2], a.shape[-1]
    #         s = a.sum().clamp_min(eps)

    #         xs = torch.arange(W, device=a.device, dtype=a.dtype)[None, :].expand(H, W)
    #         ys = torch.arange(H, device=a.device, dtype=a.dtype)[:, None].expand(H, W)
    #         cx = (a * xs).sum() / s
    #         cy = (a * ys).sum() / s
    #         return torch.stack([cx, cy], dim=0), s, a.max()

    #     def weighted_inside_gt(alpha_pred: torch.Tensor, gt_mask: torch.Tensor, eps=1e-8):
    #         a = alpha_pred.float().clamp_min(0.0)
    #         num = (a * gt_mask.float()).sum()
    #         den = a.sum().clamp_min(eps)
    #         return (num / den).item()

    #     def project_centroid_px(
    #         xyz_world: torch.Tensor,
    #         E: torch.Tensor,
    #         K: torch.Tensor,
    #         H: int,
    #         W: int,
    #         w: torch.Tensor,
    #         *,
    #         flip_y: bool = False,
    #         eps=1e-8,
    #     ):
    #         # utils3d.torch.project_cv returns uv01 in [0,1]
    #         uv01 = utils3d.torch.project_cv(xyz_world, E, K)[0]  # (N,2)
    #         m = torch.isfinite(uv01).all(1) & torch.isfinite(w) & (w > 0.01)
    #         if not m.any():
    #             return torch.tensor([float("nan"), float("nan")], device=xyz_world.device)

    #         uv01 = uv01[m]
    #         ww = w[m]

    #         uv_px = uv01 * torch.tensor([W - 1, H - 1], device=uv01.device, dtype=uv01.dtype)

    #         if flip_y:
    #             uv_px = torch.stack([uv_px[:, 0], (H - 1) - uv_px[:, 1]], dim=1)

    #         c = (uv_px * ww[:, None]).sum(0) / ww.sum().clamp_min(eps)
    #         return c
    #     def proj_centroid_area_weighted(xyz_world, rep_world, E_world, K, H, W, eps=1e-8):
    #         # camera coords to get depth
    #         ones = torch.ones((xyz_world.shape[0], 1), device=xyz_world.device, dtype=xyz_world.dtype)
    #         Xh = torch.cat([xyz_world, ones], dim=1)                # (N,4)
    #         Xc = (E_world @ Xh.t()).t()                             # (N,4) world->cam
    #         z = Xc[:, 2].clamp_min(1e-6)                            # (N,)

    #         # base opacity
    #         op = rep_world.get_opacity.view(-1).clamp_min(0.0)      # (N,)

    #         # world scales (physical)
    #         sc = rep_world.get_scaling.view(-1, 3).clamp_min(1e-9)  # (N,3)

    #         # very rough projected "area" proxy ~ (sx * sy) / z^2
    #         area = (sc[:, 0] * sc[:, 1]) / (z * z)

    #         w = op * area
    #         m = torch.isfinite(w) & (w > 0)

    #         # project
    #         uv01 = utils3d.torch.project_cv(xyz_world, E_world, K)[0]  # (N,2)
    #         m = m & torch.isfinite(uv01).all(1)

    #         if not m.any():
    #             return torch.tensor([float("nan"), float("nan")], device=xyz_world.device)

    #         uv01 = uv01[m]
    #         w = w[m]

    #         uv_px = uv01 * torch.tensor([W-1, H-1], device=uv01.device, dtype=uv01.dtype)
    #         c = (uv_px * w[:, None]).sum(0) / w.sum().clamp_min(eps)
    #         return c


    #     @torch.no_grad()
    #     def nn_err_quick(a, b, max_points=20000):
    #         """
    #         Fast-ish symmetric NN error between two point sets a,b: [N,3], [M,3].
    #         Subsamples for speed. Uses torch.cdist (OK for 20k).
    #         Returns mean NN(a->b) + mean NN(b->a).
    #         """
    #         device = a.device
    #         Na = a.shape[0]
    #         Nb = b.shape[0]
    #         if Na > max_points:
    #             idx = torch.randperm(Na, device=device)[:max_points]
    #             a = a[idx]
    #         if Nb > max_points:
    #             idx = torch.randperm(Nb, device=device)[:max_points]
    #             b = b[idx]

    #         D = torch.cdist(a, b)                # [na, nb]
    #         ab = D.min(dim=1).values.mean()
    #         ba = D.min(dim=0).values.mean()
    #         return (ab + ba).item()

    #     # @torch.no_grad()
    #     # def pose_to_world_xyz(self, rep_pose, R_row, s_aug, t_aug, c0, s_norm, c_norm):
    #     #     """
    #     #     Compute xyz_world_ref from rep_pose.get_xyz using the SAME pipeline mapping.
    #     #     """
    #     #     xyz_pose = rep_pose.get_xyz.view(-1, 3)

    #     #     R, k, b = self._pose_to_world_params(
    #     #         R_row[None], s_aug[None], t_aug[None], c0[None], s_norm[None], c_norm[None]
    #     #     )
    #     #     xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0].view(1, 3)
    #     #     return xyz_world
    #     @torch.no_grad()
    #     def pose_to_world_xyz( rep_pose,
    #                         R_row, s_aug, t_aug, c0, s_norm, c_norm):
    #         """
    #         Must match _transform_reps_for_loss:
    #         xyz_pose = rep_pose.get_xyz
    #         R,k,b = self._pose_to_world_params(...)
    #         xyz_world = (xyz_pose @ R^T) * k + b
    #         """
    #         device = rep_pose.get_xyz.device
    #         dtype  = rep_pose.get_xyz.dtype

    #         # IMPORTANT: use get_xyz (already aabb-mapped)
    #         xyz_pose = rep_pose.get_xyz.view(-1, 3)

    #         R, k, b = self._pose_to_world_params(
    #             torch.as_tensor(R_row,  device=device, dtype=dtype).unsqueeze(0),  # [1,3,3]
    #             torch.as_tensor(s_aug,  device=device, dtype=dtype).view(1),       # [1]
    #             torch.as_tensor(t_aug,  device=device, dtype=dtype).view(1,3),     # [1,3]
    #             torch.as_tensor(c0,     device=device, dtype=dtype).view(1,3),     # [1,3]
    #             torch.as_tensor(s_norm, device=device, dtype=dtype).view(1),       # [1]
    #             torch.as_tensor(c_norm, device=device, dtype=dtype).view(1,3),     # [1,3]
    #         )

    #         xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1, 1) + b[0].view(1, 3)
    #         return xyz_world
    #     # -------------------------
    #     # data sampling
    #     # -------------------------
    #     dataloader = DataLoader(
    #         copy.deepcopy(self.dataset),
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=0,
    #         collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
    #     )

    #     num_samples = 4  # force like you did

    #     gt_images = []
    #     alphas = []
    #     exts = []
    #     ints = []
    #     pose_pack = {
    #         "R_row": [], "s_aug": [], "t_aug": [], "c0": [], "s_norm": [], "c_norm": [],
    #         "origin": [], "voxel_size": []
    #     }
    #     reps = []
    #     it = iter(dataloader)
    #     uids = []

    #     for start in range(0, num_samples, batch_size):
    #         batch = min(batch_size, num_samples - start)
    #         data = next(it)
    #         args = {k: v[:batch].cuda() for k, v in data.items()}

    #         gt_images.append(args["image"] * args["alpha"][:, None])
    #         alphas.append(args["alpha"])
    #         exts.append(args["extrinsics"])
    #         ints.append(args["intrinsics"])

    #         for k in pose_pack:
    #             pose_pack[k].append(args[k])

    #         z = self.models["encoder"](args["feats"], sample_posterior=True, return_raw=False)
    #         reps_b = self.models["decoder"](z)          # list length = batch
    #         reps.extend(reps_b)
        
    #     gt_images = torch.cat(gt_images, dim=0)
    #     alphas    = torch.cat(alphas, dim=0)
    #     exts      = torch.cat(exts, dim=0)
    #     ints      = torch.cat(ints, dim=0)
    #     for k in pose_pack:
    #         pose_pack[k] = torch.cat(pose_pack[k], dim=0)

    #     # transform pose->world
    #     reps_world = self._transform_reps_for_loss(
    #         reps,
    #         R_row=pose_pack["R_row"],
    #         s_aug=pose_pack["s_aug"],
    #         t_aug=pose_pack["t_aug"],
    #         c0=pose_pack["c0"],
    #         s_norm=pose_pack["s_norm"],
    #         c_norm=pose_pack["c_norm"],
    #         origin=pose_pack["origin"],
    #         voxel_size=pose_pack["voxel_size"],
    #         intrinsics=ints,
    #         extrinsics=exts,
    #         debug=False,
    #     )

    #     # -------------------------
    #     # render twice to estimate alpha_pred
    #     # # -------------------------
    #     # H = gt_images.shape[-2]
    #     # W = gt_images.shape[-1]
    #     # self.renderer.rendering_options.resolution = W  # assumes square; if not, adapt renderer

    #     # # black bg
    #     # self.renderer.rendering_options.bg_color = (0, 0, 0)
    #     # rb = self._render_batch(reps_world, exts, ints)
    #     # cb = rb["color"]

    #     # # white bg
    #     # self.renderer.rendering_options.bg_color = (1, 1, 1)
    #     # rw = self._render_batch(reps_world, exts, ints)
    #     # cw = rw["color"]

    #     # # restore
    #     # self.renderer.rendering_options.bg_color = (0, 0, 0)

    #     # # alpha estimate
    #     # alpha_pred = 1.0 - (cw - cb).mean(dim=1)
    #     # alpha_pred = alpha_pred.clamp(0.0, 1.0)

    #     # # show the reconstructed image (black bg)
    #     # ret_dict.update({"rec_image": {"value": cb, "type": "image"}})

    #     # # -------------------------
    #     # # DEBUG block: isolate camera convention mismatch
    #     # # -------------------------
    #     # print("\n========== CAMERA/RENDERER CENTROID DEBUG ==========")

    #     # for j in range(min(num_samples, alpha_pred.shape[0])):
    #     # for j in range(num_samples):
    #     #     # 1) world-space projection debug (this should be the “correct” one)
    #     #     self.debug_camera_frame(
    #     #         rep=reps[j],
    #     #         alpha_img=alphas[j],
    #     #         E_world=exts[j],
    #     #         K=ints[j],
    #     #         R_row=pose_pack["R_row"][j],
    #     #         s_aug=pose_pack["s_aug"][j],
    #     #         t_aug=pose_pack["t_aug"][j],
    #     #         c0=pose_pack["c0"][j],
    #     #         s_norm=pose_pack["s_norm"][j],
    #     #         c_norm=pose_pack["c_norm"][j],
    #     #     )

    #     #     # 2) consistency of the transform itself
    #     #     self.debug_transform_consistency_one(reps[j], reps_world[j], j, pose_pack)
    #     #     self.diagnose_which_transform(reps[j], reps_world[j], j, pose_pack)
    #     # exit(0)

    #         # # GT alpha normalize if needed
    #         # gt_a = alphas[j]
    #         # if gt_a.max() > 1.5:
    #         #     gt_a = gt_a / 255.0
    #         # gt_a = gt_a.clamp(0.0, 1.0)

    #         # # centroids from alpha maps
    #         # c_gt, _, _ = weighted_centroid_from_alpha(gt_a)
    #         # c_r, mass_r, mx_r = weighted_centroid_from_alpha(alpha_pred[j])

    #         # inside_w = weighted_inside_gt(alpha_pred[j], gt_a > 0.1)

    #         # # build xyz_world for THIS sample from POSE rep (not reps_world)
    #         # xyz_pose = reps[j].get_xyz.view(-1, 3)

    #         # R, k, b = self._pose_to_world_params(
    #         #     pose_pack["R_row"][j:j+1],
    #         #     pose_pack["s_aug"][j:j+1],
    #         #     pose_pack["t_aug"][j:j+1],
    #         #     pose_pack["c0"][j:j+1],
    #         #     pose_pack["s_norm"][j:j+1],
    #         #     pose_pack["c_norm"][j:j+1],
    #         # )
    #         # xyz_world = (xyz_pose @ R[0].t()) * k[0].view(1) + b[0]

    #         # # weights for projection centroid
    #         # w = reps_world[j].get_opacity.view(-1)

    #         # # camera variants
    #         # E = exts[j]
    #         # K_px = ints[j]

    #         # # normalized K variant
    #         # K_n = K_px.clone()
    #         # K_n[0, 0] = K_px[0, 0] / (W - 1)
    #         # K_n[1, 1] = K_px[1, 1] / (H - 1)
    #         # K_n[0, 2] = K_px[0, 2] / (W - 1)
    #         # K_n[1, 2] = K_px[1, 2] / (H - 1)

    #         # # centered principal point variant
    #         # K_c = K_px.clone()
    #         # K_c[0, 2] = (W - 1) / 2.0
    #         # K_c[1, 2] = (H - 1) / 2.0

    #         # c_proj_px = project_centroid_px(xyz_world, E, K_px, H, W, w, flip_y=False)
    #         # c_proj_n  = project_centroid_px(xyz_world, E, K_n,  H, W, w, flip_y=False)
    #         # c_proj_px_fy = project_centroid_px(xyz_world, E, K_px, H, W, w, flip_y=True)
    #         # c_proj_c  = project_centroid_px(xyz_world, E, K_c,  H, W, w, flip_y=False)

    #         # # compare to rendered centroid
    #         # def l2(a, b):
    #         #     if torch.isnan(a).any() or torch.isnan(b).any():
    #         #         return float("nan")
    #         #     d = (a - b).float()
    #         #     return float(torch.sqrt((d * d).sum()).item())

    #         # d_px = l2(c_proj_px, c_r)
    #         # d_n  = l2(c_proj_n,  c_r)
    #         # d_fy = l2(c_proj_px_fy, c_r)
    #         # d_c  = l2(c_proj_c,  c_r)

    #         # print(f"\n[j={j}]")
    #         # print(f"  GT alpha centroid      : {c_gt.tolist()}")
    #         # print(f"  RENDER alpha centroid  : {c_r.tolist()}   (mass={float(mass_r):.3e}, max={float(mx_r):.3f}, insideGT_w={inside_w:.4f})")

    #         # print(f"  proj centroid K_px     : {c_proj_px.tolist()}   dist_to_render={d_px:.2f}px")
    #         # print(f"  proj centroid K_norm   : {c_proj_n.tolist()}    dist_to_render={d_n:.2f}px")
    #         # print(f"  proj centroid K_px+Yflip: {c_proj_px_fy.tolist()} dist_to_render={d_fy:.2f}px")
    #         # print(f"  proj centroid K_center : {c_proj_c.tolist()}    dist_to_render={d_c:.2f}px")

    #         # # quick winner
    #         # dists = {"K_px": d_px, "K_norm": d_n, "K_px+Yflip": d_fy, "K_center": d_c}
    #         # winner = min(dists, key=lambda k: (np.inf if np.isnan(dists[k]) else dists[k]))
    #         # print(f"  ==> best match to renderer: {winner} (dist {dists[winner]:.2f}px)")

    #         # c_proj_area = proj_centroid_area_weighted(
    #         #     xyz_world,
    #         #     reps_world[j],
    #         #     exts[j],
    #         #     ints[j],
    #         #     H, W
    #         # )
    #         # print(f"[proj-centroid areaW j={j}] {c_proj_area.tolist()}  delta_to_render={(c_proj_area - c_r).tolist()}")

    #         # coords_j = args['feats'].coords[args['feats'].coords[:,0]==j][:,1:]  # YOU must adapt this line to your dataset field
    #         # self.debug_gt_alpha_vs_camera(
    #         #     coords_ijk=coords_j,
    #         #     alpha_gt=alphas[j],
    #         #     E_world=exts[j],
    #         #     K=ints[j],
    #         #     R_row=pose_pack["R_row"][j],
    #         #     s_aug=pose_pack["s_aug"][j],
    #         #     t_aug=pose_pack["t_aug"][j],
    #         #     c0=pose_pack["c0"][j],
    #         #     s_norm=pose_pack["s_norm"][j],
    #         #     c_norm=pose_pack["c_norm"][j],
    #         #     res=int(self.models["decoder"].resolution),
    #         # )


    #     print("\n===================================================\n")

    #     # -------------------------
    #     # multiview render (unchanged)
    #     # -------------------------
    #     self.renderer.rendering_options.resolution = 512

    #     yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    #     yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
    #     yaws = [y + yaws_offset for y in yaws]
    #     pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

    #     miltiview_images = []
    #     for yaw, pitch in zip(yaws, pitch):
    #         orig = torch.tensor([
    #             np.sin(yaw) * np.cos(pitch),
    #             np.cos(yaw) * np.cos(pitch),
    #             np.sin(pitch),
    #         ]).float().cuda() * 2
    #         fov = torch.deg2rad(torch.tensor(30)).cuda()
    #         extrinsics = utils3d.torch.extrinsics_look_at(
    #             orig,
    #             torch.tensor([0, 0, 0]).float().cuda(),
    #             torch.tensor([0, 0, 1]).float().cuda(),
    #         )
    #         intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    #         extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
    #         intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)

    #         render_results = self._render_batch(reps_world, extrinsics, intrinsics)
    #         miltiview_images.append(render_results["color"])

    #     miltiview_images = torch.cat([
    #         torch.cat(miltiview_images[:2], dim=-2),
    #         torch.cat(miltiview_images[2:], dim=-2),
    #     ], dim=-1)

    #     ret_dict.update({"miltiview_image": {"value": miltiview_images, "type": "image"}})

    #     self.renderer.rendering_options.bg_color = "random"
    #     return ret_dict

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
            "origin": [], "voxel_size": []
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
