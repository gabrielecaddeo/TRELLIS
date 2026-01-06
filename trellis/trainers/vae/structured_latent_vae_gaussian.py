from typing import *
import copy
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
def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm):
    """
    Returns per-sample (R, k, b) so that:
      x_w = (x_pose @ R.T) * k + b
    Shapes:
      R: (B,3,3)
      k: (B,1,1)
      b: (B,1,3)
    """
    # ensure float
    R = R_row
    k = 1.0 / (s_aug * s_norm)                  # (B,)
    k = k.view(-1, 1, 1)
    b = c_norm + c0 / s_norm.unsqueeze(-1) - (t_aug @ R.transpose(1, 2)) * k.squeeze(-1)  # (B,3)
    b = b.view(-1, 1, 3)
    return R, k, b

def _transform_reps_for_loss(
    self,
    reps,
    *,
    R_row, s_aug, t_aug, c0, s_norm, c_norm,
    posed_mask=None
):
    """
    reps: list[Gaussian] length B
    returns list[Gaussian] length B, transformed into render/world frame.
    """
    B = len(reps)
    device = reps[0]._xyz.device
    dtype  = reps[0]._xyz.dtype

    R_row  = R_row.to(device=device, dtype=dtype)
    s_aug  = s_aug.to(device=device, dtype=dtype)
    t_aug  = t_aug.to(device=device, dtype=dtype)
    c0     = c0.to(device=device, dtype=dtype)
    s_norm = s_norm.to(device=device, dtype=dtype)
    c_norm = c_norm.to(device=device, dtype=dtype)

    if posed_mask is not None:
        posed_mask = posed_mask.to(device=device)

    R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)

    act = self.models["decoder"].rep_config.get("scaling_activation", "exp")

    out = []
    for i in range(B):
        rep = reps[i]
        rep2 = copy.copy(rep)

        if posed_mask is not None and not bool(posed_mask[i].item()):
            out.append(rep2)
            continue

        Ri = R[i]    # (3,3)
        ki = k[i]    # (1,1)
        bi = b[i]    # (1,3)

        # --- xyz ---
        xyz = rep2._xyz.view(-1, 3)                      # (Ng,3)
        xyz_w = (xyz @ Ri.t()) * ki + bi                 # (Ng,3)
        rep2._xyz = xyz_w

        # --- rotation (quat) ---
        q = rep2._rotation
        if q is not None and q.shape[-1] == 4:
            qR = self._quat_from_rotmat(Ri).view(1, 4).expand_as(q)
            rep2._rotation = F.normalize(self._quat_mul(qR, q), dim=-1)

        # --- scaling ---
        # use activated scaling, scale it, then invert activation back to raw
        s = rep2.get_scaling                              # (Ng,3) activated
        s_new = s * ki.squeeze()                          # uniform
        rep2._scaling = self._inv_scaling_activation(s_new, act)

        out.append(rep2)

    return out

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
        R_row: torch.Tensor,
        s_aug: torch.Tensor,
        t_aug: torch.Tensor,
        c0: torch.Tensor,
        s_norm: torch.Tensor,
        c_norm: torch.Tensor,
        posed_mask: torch.Tensor = None,
        return_aux: bool = False,
        **kwargs
    ): 
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
        # align ONLY for rendering loss
        reps_world = self._transform_reps_for_loss(
            reps,
            R_row=R_row, s_aug=s_aug, t_aug=t_aug, c0=c0, s_norm=s_norm, c_norm=c_norm,
            posed_mask=posed_mask,
        )

        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps_world, extrinsics, intrinsics)     
        
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
        pose_pack = { "R_row": [], "s_aug": [], "t_aug": [], "c0": [], "s_norm": [], "c_norm": [] }
        reps = []

        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items()}

            gt_images.append(args['image'] * args['alpha'][:, None])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])

            for k in pose_pack:
                pose_pack[k].append(args[k])

            z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
            reps.extend(self.models['decoder'](z))

        gt_images = torch.cat(gt_images, dim=0)
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        for k in pose_pack:
            pose_pack[k] = torch.cat(pose_pack[k], dim=0)
        reps_world = self._transform_reps_for_loss(
            reps,
            R_row=pose_pack["R_row"],
            s_aug=pose_pack["s_aug"],
            t_aug=pose_pack["t_aug"],
            c0=pose_pack["c0"],
            s_norm=pose_pack["s_norm"],
            c_norm=pose_pack["c_norm"],
        )

        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps_world, exts, ints)
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
            render_results = self._render_batch(reps_world, extrinsics, intrinsics)
            miltiview_images.append(render_results['color'])

        ## Concatenate views
        miltiview_images = torch.cat([
            torch.cat(miltiview_images[:2], dim=-2),
            torch.cat(miltiview_images[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
