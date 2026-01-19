from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from ..basic import BasicTrainer
from ...representations import MeshExtractResult
from ...renderers import MeshRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, smooth_l1_loss, ssim, lpips
from ...utils.data_utils import recursive_to_device


class SLatVaeMeshDecoderTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE Mesh Decoder.
    
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
    """
    
    def __init__(
        self,
        *args,
        depth_loss_type: str = 'l1',
        lambda_depth: int = 1,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_tsdf: float = 0.01,
        lambda_color: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_loss_type = depth_loss_type
        self.lambda_depth = lambda_depth
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_tsdf = lambda_tsdf
        self.lambda_color = lambda_color
        self.use_color = self.lambda_color > 0
        
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                             "far" : 3}
        self.renderer = MeshRenderer(rendering_options, device=self.device)
        
    def _render_batch(self, reps: List[MeshExtractResult], extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                      return_types=['mask', 'normal', 'depth']) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
        Returns: 
            a dict with
                reg_loss : [N] tensor of regularization losses
                mask : [N x 1 x H x W] tensor of rendered masks
                normal : [N x 3 x H x W] tensor of rendered normals
                depth : [N x 1 x H x W] tensor of rendered depths
        """
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types)
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    @staticmethod
    def _tsdf_reg_loss(rep: MeshExtractResult, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        # Calculate tsdf
        with torch.no_grad():
            # Project points to camera and calculate pseudo-sdf as difference between gt depth and projected depth
            projected_pts, pts_depth = utils3d.torch.project_cv(extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v)
            projected_pts = (projected_pts - 0.5) * 2.0
            depth_map_res = depth_map.shape[1]
            gt_depth = torch.nn.functional.grid_sample(depth_map.reshape(1, 1, depth_map_res, depth_map_res), 
            projected_pts.reshape(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=True)
            pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
            # Truncate pseudo-sdf
            delta = 1 / rep.res * 3.0
            trunc_mask = pseudo_sdf > -delta
        
        # Loss
        gt_tsdf = pseudo_sdf[trunc_mask]
        tsdf = rep.tsdf_s.flatten()[trunc_mask]
        gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
        return torch.mean((tsdf - gt_tsdf) ** 2)
    
    def _calc_tsdf_loss(self, reps : list[MeshExtractResult], depth_maps, extrinsics, intrinsics) -> torch.Tensor:
        tsdf_loss = 0.0
        for i, rep in enumerate(reps):
            tsdf_loss += self._tsdf_reg_loss(rep, depth_maps[i], extrinsics[i], intrinsics[i])
        return tsdf_loss / len(reps)
    
    @torch.no_grad()
    def _flip_normal(self, normal: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Flip normal to align with camera.
        """
        normal = normal * 2.0 - 1.0
        R = torch.zeros_like(extrinsics)
        R[:, :3, :3] = extrinsics[:, :3, :3]
        R[:, 3, 3] = 1.0
        view_dir = utils3d.torch.unproject_cv(
            utils3d.torch.image_uv(*normal.shape[-2:], device=self.device).reshape(1, -1, 2),
            torch.ones(*normal.shape[-2:], device=self.device).reshape(1, -1),
            R, intrinsics
        ).reshape(-1, *normal.shape[-2:], 3).permute(0, 3, 1, 2)
        unflip = (normal * view_dir).sum(1, keepdim=True) < 0
        normal *= unflip * 2.0 - 1.0
        return (normal + 1.0) / 2.0
    
    def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
        """
        Combination of L1, SSIM, and LPIPS loss.
        """
        if gt.shape[1] != 3:
            assert gt.shape[-1] == 3
            gt = gt.permute(0, 3, 1, 2)
        if pred.shape[1] != 3:
            assert pred.shape[-1] == 3
            pred = pred.permute(0, 3, 1, 2)
        terms = {
            f"{name}_loss" : l1_loss(gt, pred),
            f"{name}_loss_ssim" : 1 - ssim(gt, pred),
            f"{name}_loss_lpips" : lpips(gt, pred)
        }
        terms[f"{name}_loss_perceptual"] = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim + terms[f"{name}_loss_lpips"] * self.lambda_lpips
        return terms
    
    def geometry_losses(
        self,
        reps: List[MeshExtractResult],
        mesh: List[Dict],
        normal_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        with torch.no_grad():
            gt_meshes = []
            for i in range(len(reps)):
                gt_mesh = MeshExtractResult(mesh[i]['vertices'].to(self.device), mesh[i]['faces'].to(self.device))
                gt_meshes.append(gt_mesh)
            target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=['mask', 'depth', 'normal'])
            target['normal'] = self._flip_normal(target['normal'], extrinsics, intrinsics)
                
        terms = edict(geo_loss = 0.0)
        if self.lambda_tsdf > 0:
            tsdf_loss = self._calc_tsdf_loss(reps, target['depth'], extrinsics, intrinsics)
            terms['tsdf_loss'] = tsdf_loss
            terms['geo_loss'] += tsdf_loss * self.lambda_tsdf
        
        return_types = ['mask', 'depth', 'normal', 'normal_map'] if self.use_color else ['mask', 'depth', 'normal']
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
        
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            for k, v in buffer.items():
                buffer[k] = v[success_mask]
            for k, v in target.items():
                target[k] = v[success_mask]
            
            terms['mask_loss'] = l1_loss(buffer['mask'], target['mask']) 
            if self.depth_loss_type == 'l1':
                terms['depth_loss'] = l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'])
            elif self.depth_loss_type == 'smooth_l1':
                terms['depth_loss'] = smooth_l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'], beta=1.0 / (2 * reps[0].res))
            else:
                raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
            terms.update(self._perceptual_loss(buffer['normal'] * target['mask'], target['normal'] * target['mask'], 'normal'))
            terms['geo_loss'] = terms['geo_loss'] + terms['mask_loss'] + terms['depth_loss'] * self.lambda_depth + terms['normal_loss_perceptual']
            if self.use_color and normal_map is not None:
                terms.update(self._perceptual_loss(normal_map[success_mask], buffer['normal_map'], 'normal_map'))
                terms['geo_loss'] = terms['geo_loss'] + terms['normal_map_loss_perceptual'] * self.lambda_color
                
        return terms
      
    def color_losses(self, reps, image, alpha, extrinsics, intrinsics):
        terms = edict(color_loss = torch.tensor(0.0, device=self.device))
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=['color'])
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            terms.update(self._perceptual_loss((image * alpha[:, None])[success_mask], buffer['color'][success_mask], 'color'))
            terms['color_loss'] = terms['color_loss'] + terms['color_loss_perceptual'] * self.lambda_color
        return terms
    
    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        mesh: List[Dict],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        normal_map: torch.Tensor = None,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            latents: The [N x * x C] sparse latents
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            mesh: The list of dictionaries of meshes.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        reps = self.training_models['decoder'](latents)
        self.renderer.rendering_options.resolution = image.shape[-1]
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        terms['reg_loss'] = sum([rep.reg_loss for rep in reps]) / len(reps)
        terms['loss'] = terms['loss'] + terms['reg_loss']
        
        geo_terms = self.geometry_losses(reps, mesh, normal_map, extrinsics, intrinsics)
        terms.update(geo_terms)
        terms['loss'] = terms['loss'] + terms['geo_loss']
                
        if self.use_color:
            color_terms = self.color_losses(reps, image, alpha, extrinsics, intrinsics)
            terms.update(color_terms)
            terms['loss'] = terms['loss'] + terms['color_loss']
             
        return terms, {}
    
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
        gt_normal_maps = []
        gt_meshes = []
        exts = []
        ints = []
        reps = []
        num_samples = 4
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = recursive_to_device(data, 'cuda')
            gt_images.append(args['image'] * args['alpha'][:, None])
            if self.use_color and 'normal_map' in data:
                gt_normal_maps.append(args['normal_map'])
            gt_meshes.extend(args['mesh'])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            reps.extend(self.models['decoder'](args['latents']))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        if self.use_color and gt_normal_maps:
            gt_normal_maps = torch.cat(gt_normal_maps, dim=0)
            ret_dict.update({f'gt_normal_map': {'value': gt_normal_maps, 'type': 'image'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        gt_render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts, ints, return_types=['normal'])
        ret_dict.update({f'gt_normal': {'value': self._flip_normal(gt_render_results['normal'], exts, ints), 'type': 'image'}})
        return_types = ['normal']
        if self.use_color:
            return_types.append('color')
            if 'normal_map' in data:
                return_types.append('normal_map')
        render_results = self._render_batch(reps, exts, ints, return_types=return_types)
        ret_dict.update({f'rec_normal': {'value': render_results['normal'], 'type': 'image'}})
        if 'color' in return_types:
            ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})
        if 'normal_map' in return_types:
            ret_dict.update({f'rec_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        multiview_normals = []
        multiview_normal_maps = []
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
            render_results = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
            multiview_normals.append(render_results['normal'])
            if 'color' in return_types:
                miltiview_images.append(render_results['color'])
            if 'normal_map' in return_types:
                multiview_normal_maps.append(render_results['normal_map'])

        ## Concatenate views
        multiview_normals = torch.cat([
            torch.cat(multiview_normals[:2], dim=-2),
            torch.cat(multiview_normals[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'multiview_normal': {'value': multiview_normals, 'type': 'image'}})
        if 'color' in return_types:
            miltiview_images = torch.cat([
                torch.cat(miltiview_images[:2], dim=-2),
                torch.cat(miltiview_images[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_image': {'value': miltiview_images, 'type': 'image'}})
        if 'normal_map' in return_types:
            multiview_normal_maps = torch.cat([
                torch.cat(multiview_normal_maps[:2], dim=-2),
                torch.cat(multiview_normal_maps[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_normal_map': {'value': multiview_normal_maps, 'type': 'image'}})
                            
        return ret_dict



# class SLatVaeMeshDecoderTrainerPose(BasicTrainer):
#     """
#     Trainer for structured latent VAE Mesh Decoder.
    
#     Args:
#         models (dict[str, nn.Module]): Models to train.
#         dataset (torch.utils.data.Dataset): Dataset.
#         output_dir (str): Output directory.
#         load_dir (str): Load directory.
#         step (int): Step to load.
#         batch_size (int): Batch size.
#         batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
#         batch_split (int): Split batch with gradient accumulation.
#         max_steps (int): Max steps.
#         optimizer (dict): Optimizer config.
#         lr_scheduler (dict): Learning rate scheduler config.
#         elastic (dict): Elastic memory management config.
#         grad_clip (float or dict): Gradient clip config.
#         ema_rate (float or list): Exponential moving average rates.
#         fp16_mode (str): FP16 mode.
#             - None: No FP16.
#             - 'inflat_all': Hold a inflated fp32 master param for all params.
#             - 'amp': Automatic mixed precision.
#         fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
#         finetune_ckpt (dict): Finetune checkpoint.
#         log_param_stats (bool): Log parameter stats.
#         i_print (int): Print interval.
#         i_log (int): Log interval.
#         i_sample (int): Sample interval.
#         i_save (int): Save interval.
#         i_ddpcheck (int): DDP check interval.
        
#         loss_type (str): Loss type. Can be 'l1', 'l2'
#         lambda_ssim (float): SSIM loss weight.
#         lambda_lpips (float): LPIPS loss weight.
#     """
    
#     def __init__(
#         self,
#         *args,
#         depth_loss_type: str = 'l1',
#         lambda_depth: int = 1,
#         lambda_ssim: float = 0.2,
#         lambda_lpips: float = 0.2,
#         lambda_tsdf: float = 0.01,
#         lambda_color: float = 0.1,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.depth_loss_type = depth_loss_type
#         self.lambda_depth = lambda_depth
#         self.lambda_ssim = lambda_ssim
#         self.lambda_lpips = lambda_lpips
#         self.lambda_tsdf = lambda_tsdf
#         self.lambda_color = lambda_color
#         self.use_color = self.lambda_color > 0
        
#         self._init_renderer()
        
#     def _init_renderer(self):
#         rendering_options = {"near" : 1,
#                              "far" : 3}
#         self.renderer = MeshRenderer(rendering_options, device=self.device)
        
#     def _render_batch(self, reps: List[MeshExtractResult], extrinsics: torch.Tensor, intrinsics: torch.Tensor,
#                       return_types=['mask', 'normal', 'depth']) -> Dict[str, torch.Tensor]:
#         """
#         Render a batch of representations.

#         Args:
#             reps: The dictionary of lists of representations.
#             extrinsics: The [N x 4 x 4] tensor of extrinsics.
#             intrinsics: The [N x 3 x 3] tensor of intrinsics.
#             return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
#         Returns: 
#             a dict with
#                 reg_loss : [N] tensor of regularization losses
#                 mask : [N x 1 x H x W] tensor of rendered masks
#                 normal : [N x 3 x H x W] tensor of rendered normals
#                 depth : [N x 1 x H x W] tensor of rendered depths
#         """
#         ret = {k : [] for k in return_types}
#         for i, rep in enumerate(reps):
#             out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types)
#             for k in out_dict:
#                 ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
#         for k in ret:
#             ret[k] = torch.stack(ret[k])
#         return ret
    
#     @staticmethod
#     def _tsdf_reg_loss(rep: MeshExtractResult, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
#         # Calculate tsdf
#         with torch.no_grad():
#             # Project points to camera and calculate pseudo-sdf as difference between gt depth and projected depth
#             projected_pts, pts_depth = utils3d.torch.project_cv(extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v)
#             projected_pts = (projected_pts - 0.5) * 2.0
#             depth_map_res = depth_map.shape[1]
#             gt_depth = torch.nn.functional.grid_sample(depth_map.reshape(1, 1, depth_map_res, depth_map_res), 
#             projected_pts.reshape(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=True)
#             pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
#             # Truncate pseudo-sdf
#             delta = 1 / rep.res * 3.0
#             trunc_mask = pseudo_sdf > -delta
        
#         # Loss
#         gt_tsdf = pseudo_sdf[trunc_mask]
#         tsdf = rep.tsdf_s.flatten()[trunc_mask]
#         gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
#         return torch.mean((tsdf - gt_tsdf) ** 2)
    
#     def _calc_tsdf_loss(self, reps : list[MeshExtractResult], depth_maps, extrinsics, intrinsics) -> torch.Tensor:
#         tsdf_loss = 0.0
#         for i, rep in enumerate(reps):
#             tsdf_loss += self._tsdf_reg_loss(rep, depth_maps[i], extrinsics[i], intrinsics[i])
#         return tsdf_loss / len(reps)
    
#     @torch.no_grad()
#     def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm, eps=1e-12):
#         B = R_row.shape[0]
#         assert R_row.shape == (B,3,3)
#         assert t_aug.shape == (B,3)
#         assert c0.shape == (B,3)
#         assert c_norm.shape == (B,3)

#         s_aug  = s_aug.view(B, 1)
#         s_norm = s_norm.view(B, 1)

#         k = 1.0 / (s_aug.clamp_min(eps) * s_norm.clamp_min(eps))     # (B,1)
#         tR = torch.einsum("bi,bij->bj", t_aug, R_row.transpose(1,2)) # (B,3)
#         b  = c_norm + (c0 / s_norm.clamp_min(eps)) - (tR * k)        # (B,3)

#         return R_row, k, b


    
    
#     # def _transform_mesh_reps_for_loss(
#     #     self,
#     #     reps,                      # list[MeshExtractResult], len=B
#     #     *,
#     #     R_row, s_aug, t_aug, c0,
#     #     s_norm, c_norm,
#     #     posed_mask=None,
#     #     coord_scale: float = 1.0,  # set to 2.0 if you changed get_defomed_verts to unit2
#     # ):
#     #     """
#     #     Transform decoded mesh reps from POSE space -> WORLD space.

#     #     Applies to:
#     #     - rep.vertices
#     #     - rep.tsdf_v (if present)
#     #     - (optionally normals/attrs only if they encode positions; usually they don't)

#     #     posed_mask: [B] bool, if provided, only transform those True; others pass-through.
#     #     """
#     #     B = len(reps)
#     #     device = reps[0].vertices.device

#     #     # promote pose params to tensors on device
#     #     R_row  = R_row.to(device)
#     #     s_aug  = s_aug.to(device)
#     #     t_aug  = t_aug.to(device)
#     #     c0     = c0.to(device)
#     #     s_norm = s_norm.to(device)
#     #     c_norm = c_norm.to(device)

#     #     R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)  # R:[B,3,3], k:[B,1], b:[B,3]

#     #     out = []
#     #     for i in range(B):
#     #         rep = reps[i]

#     #         if posed_mask is not None and not bool(posed_mask[i].item()):
#     #             out.append(rep)
#     #             continue

#     #         V = rep.vertices
#     #         if i == 0:
#     #             print("IN _transform_mesh_reps_for_loss shapes:",
#     #                 "R_row", tuple(R_row.shape),
#     #                 "s_aug", tuple(s_aug.shape),
#     #                 "t_aug", tuple(t_aug.shape),
#     #                 "c0", tuple(c0.shape),
#     #                 "s_norm", tuple(s_norm.shape),
#     #                 "c_norm", tuple(c_norm.shape))

#     #             # also print per-sample shapes just in case something is ragged
#     #             print("per-sample:",
#     #                 "t_aug[i]", tuple(t_aug[i].shape),
#     #                 "c0[i]", tuple(c0[i].shape),
#     #                 "c_norm[i]", tuple(c_norm[i].shape))
#     #             print("[DBG] b shape", tuple(b.shape))
#     #         # row-vector transform:
#     #         Vw = (V @ R[i].t()) * k[i].view(1, 1) + b[i].view(1, 3)
#     #         Vw = Vw * coord_scale

#     #         rep_w = MeshExtractResult(
#     #             vertices=Vw,
#     #             faces=rep.faces,
#     #             vertex_attrs=rep.vertex_attrs,
#     #             res=rep.res,
#     #         )

#     #         # training-only fields
#     #         rep_w.success = rep.success
#     #         rep_w.reg_loss = rep.reg_loss

#     #         if rep.tsdf_v is not None:
#     #             tv = rep.tsdf_v
#     #             tvw = (tv @ R[i].t()) * k[i].view(1, 1) + b[i].view(1, 3)
#     #             rep_w.tsdf_v = tvw * coord_scale
#     #             rep_w.tsdf_s = rep.tsdf_s

#     #         out.append(rep_w)

#     #     return out
#     def _transform_mesh_reps_for_loss(
#         self, reps, *,
#         R_row, s_aug, t_aug, c0, s_norm, c_norm,
#         posed_mask=None,
#         assume_mesh_in_unit05=True,  # vertices ~[-0.5,0.5]
#     ):
#         B = len(reps)
#         device = reps[0].vertices.device

#         R_row  = R_row.to(device)
#         s_aug  = s_aug.to(device)
#         t_aug  = t_aug.to(device)
#         c0     = c0.to(device)
#         s_norm = s_norm.to(device)
#         c_norm = c_norm.to(device)

#         R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)
#         print("k stats:", float(k.min()), float(k.max()))
#         print("b norm:", b.norm(dim=1))
#         out = []
#         for i in range(B):
#             rep = reps[i]
#             if posed_mask is not None and not bool(posed_mask[i].item()):
#                 out.append(rep)
#                 continue

#             V = rep.vertices
#             V_pose = (2.0 * V) if assume_mesh_in_unit05 else V

#             Vw = (V_pose @ R[i].t()) * k[i].view(1,1) + b[i].view(1,3)

#             rep_w = MeshExtractResult(vertices=Vw, faces=rep.faces, vertex_attrs=rep.vertex_attrs, res=rep.res)
#             rep_w.success = rep.success
#             rep_w.reg_loss = rep.reg_loss

#             if rep.tsdf_v is not None:
#                 tv = rep.tsdf_v
#                 tv_pose = (2.0 * tv) if assume_mesh_in_unit05 else tv
#                 rep_w.tsdf_v = (tv_pose @ R[i].t()) * k[i].view(1,1) + b[i].view(1,3)
#                 rep_w.tsdf_s = rep.tsdf_s

#             out.append(rep_w)

#         return out



#     @torch.no_grad()
#     def _flip_normal(self, normal: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
#         """
#         Flip normal to align with camera.
#         """
#         normal = normal * 2.0 - 1.0
#         R = torch.zeros_like(extrinsics)
#         R[:, :3, :3] = extrinsics[:, :3, :3]
#         R[:, 3, 3] = 1.0
#         view_dir = utils3d.torch.unproject_cv(
#             utils3d.torch.image_uv(*normal.shape[-2:], device=self.device).reshape(1, -1, 2),
#             torch.ones(*normal.shape[-2:], device=self.device).reshape(1, -1),
#             R, intrinsics
#         ).reshape(-1, *normal.shape[-2:], 3).permute(0, 3, 1, 2)
#         unflip = (normal * view_dir).sum(1, keepdim=True) < 0
#         normal *= unflip * 2.0 - 1.0
#         return (normal + 1.0) / 2.0
    
#     def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
#         """
#         Combination of L1, SSIM, and LPIPS loss.
#         """
#         if gt.shape[1] != 3:
#             assert gt.shape[-1] == 3
#             gt = gt.permute(0, 3, 1, 2)
#         if pred.shape[1] != 3:
#             assert pred.shape[-1] == 3
#             pred = pred.permute(0, 3, 1, 2)
#         terms = {
#             f"{name}_loss" : l1_loss(gt, pred),
#             f"{name}_loss_ssim" : 1 - ssim(gt, pred),
#             f"{name}_loss_lpips" : lpips(gt, pred)
#         }
#         terms[f"{name}_loss_perceptual"] = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim + terms[f"{name}_loss_lpips"] * self.lambda_lpips
#         return terms
    
#     def geometry_losses(
#         self,
#         reps: List[MeshExtractResult],
#         mesh: List[Dict],
#         normal_map: torch.Tensor,
#         extrinsics: torch.Tensor,
#         intrinsics: torch.Tensor,
#     ):
#         with torch.no_grad():
#             gt_meshes = []
#             for i in range(len(reps)):
#                 gt_mesh = MeshExtractResult(mesh[i]['vertices'].to(self.device), mesh[i]['faces'].to(self.device))
#                 gt_meshes.append(gt_mesh)
#             target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=['mask', 'depth', 'normal'])
#             target['normal'] = self._flip_normal(target['normal'], extrinsics, intrinsics)
                
#         terms = edict(geo_loss = 0.0)
#         if self.lambda_tsdf > 0:
#             tsdf_loss = self._calc_tsdf_loss(reps, target['depth'], extrinsics, intrinsics)
#             terms['tsdf_loss'] = tsdf_loss
#             terms['geo_loss'] += tsdf_loss * self.lambda_tsdf
        
#         return_types = ['mask', 'depth', 'normal', 'normal_map'] if self.use_color else ['mask', 'depth', 'normal']
#         buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
        
#         success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
#         if success_mask.sum() != 0:
#             for k, v in buffer.items():
#                 buffer[k] = v[success_mask]
#             for k, v in target.items():
#                 target[k] = v[success_mask]
            
#             terms['mask_loss'] = l1_loss(buffer['mask'], target['mask']) 
#             if self.depth_loss_type == 'l1':
#                 terms['depth_loss'] = l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'])
#             elif self.depth_loss_type == 'smooth_l1':
#                 terms['depth_loss'] = smooth_l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'], beta=1.0 / (2 * reps[0].res))
#             else:
#                 raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
#             terms.update(self._perceptual_loss(buffer['normal'] * target['mask'], target['normal'] * target['mask'], 'normal'))
#             terms['geo_loss'] = terms['geo_loss'] + terms['mask_loss'] + terms['depth_loss'] * self.lambda_depth + terms['normal_loss_perceptual']
#             if self.use_color and normal_map is not None:
#                 terms.update(self._perceptual_loss(normal_map[success_mask], buffer['normal_map'], 'normal_map'))
#                 terms['geo_loss'] = terms['geo_loss'] + terms['normal_map_loss_perceptual'] * self.lambda_color
                
#         return terms
      
#     def color_losses(self, reps, image, alpha, extrinsics, intrinsics):
#         terms = edict(color_loss = torch.tensor(0.0, device=self.device))
#         buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=['color'])
#         success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
#         if success_mask.sum() != 0:
#             terms.update(self._perceptual_loss((image * alpha[:, None])[success_mask], buffer['color'][success_mask], 'color'))
#             terms['color_loss'] = terms['color_loss'] + terms['color_loss_perceptual'] * self.lambda_color
#         return terms
    
#     def training_losses(
#         self,
#         latents: SparseTensor,
#         image: torch.Tensor,
#         alpha: torch.Tensor,
#         mesh: List[Dict],
#         extrinsics: torch.Tensor,
#         intrinsics: torch.Tensor,
#         R_row: torch.Tensor,          # [B,3,3]
#         s_aug: torch.Tensor,          # [B]
#         t_aug: torch.Tensor,          # [B,3]
#         c0: torch.Tensor,             # [B,3]
#         s_norm: torch.Tensor,         # [B]
#         c_norm: torch.Tensor,         # [B,3]
#         origin: torch.Tensor = None,
#         voxel_size: torch.Tensor = None,
#         posed_mask: torch.Tensor = None,  # [B] bool
#         normal_map: torch.Tensor = None,
#     ) -> Tuple[Dict, Dict]:
#         """
#         Compute training losses.

#         Args:
#             latents: The [N x * x C] sparse latents
#             image: The [N x 3 x H x W] tensor of images.
#             alpha: The [N x H x W] tensor of alpha channels.
#             mesh: The list of dictionaries of meshes.
#             extrinsics: The [N x 4 x 4] tensor of extrinsics.
#             intrinsics: The [N x 3 x 3] tensor of intrinsics.

#         Returns:
#             a dict with the key "loss" containing a scalar tensor.
#             may also contain other keys for different terms.
#         """
#         reps = self.training_models['decoder'](latents)
#         reps_world = self._transform_mesh_reps_for_loss(
#         reps,
#         R_row=R_row, s_aug=s_aug, t_aug=t_aug, c0=c0,
#         s_norm=s_norm, c_norm=c_norm,
#         posed_mask=posed_mask,
#     )

#         self.renderer.rendering_options.resolution = image.shape[-1]
        
#         terms = edict(loss = 0.0, rec = 0.0)
        
#         terms['reg_loss'] = sum([rep.reg_loss for rep in reps_world]) / len(reps_world)
#         terms['loss'] = terms['loss'] + terms['reg_loss']
        
#         geo_terms = self.geometry_losses(reps_world, mesh, normal_map, extrinsics, intrinsics)
#         terms.update(geo_terms)
#         terms['loss'] = terms['loss'] + terms['geo_loss']
                
#         if self.use_color:
#             color_terms = self.color_losses(reps_world, image, alpha, extrinsics, intrinsics)
#             terms.update(color_terms)
#             terms['loss'] = terms['loss'] + terms['color_loss']
             
#         return terms, {}
    
#     @torch.no_grad()
#     def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool=False) -> Dict:
#         dataloader = DataLoader(
#             copy.deepcopy(self.dataset),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0,
#             collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
#             drop_last=True,   # important for simplicity
#         )

#         num_samples = 4
#         batch_size = min(batch_size, num_samples)

#         ret_dict = {}
#         gt_images, gt_normal_maps = [], []
#         gt_meshes, exts, ints = [], [], []
#         reps_all = []

#         pose_pack = {k: [] for k in ["R_row","s_aug","t_aug","c0","s_norm","c_norm","origin","voxel_size","posed_mask"]}

#         it = iter(dataloader)
#         n_done = 0
#         while n_done < num_samples:
#             data = next(it)
#             args = recursive_to_device(data, 'cuda')

#             B = args["image"].shape[0]
#             take = min(B, num_samples - n_done)

#             gt_images.append((args["image"][:take] * args["alpha"][:take, None]))
#             gt_meshes.extend(args["mesh"][:take])
#             exts.append(args["extrinsics"][:take])
#             ints.append(args["intrinsics"][:take])

#             reps_b = self.models["decoder"](args["latents"])  # list length B
#             reps_all.extend(reps_b[:take])

#             for k in pose_pack:
#                 pose_pack[k].append(args[k][:take])

#             n_done += take

#         gt_images = torch.cat(gt_images, 0)
#         exts = torch.cat(exts, 0)
#         ints = torch.cat(ints, 0)
#         for k in pose_pack:
#             pose_pack[k] = torch.cat(pose_pack[k], 0)

#         ret_dict["gt_image"] = {"value": gt_images, "type": "image"}
#         print("shapes:",
#         pose_pack["R_row"].shape,
#         pose_pack["t_aug"].shape,
#         pose_pack["c0"].shape,
#         pose_pack["c_norm"].shape)
        
#         reps_world = self._transform_mesh_reps_for_loss(
#             reps_all,
#             R_row=pose_pack["R_row"],
#             s_aug=pose_pack["s_aug"],
#             t_aug=pose_pack["t_aug"],
#             c0=pose_pack["c0"],
#             s_norm=pose_pack["s_norm"],
#             c_norm=pose_pack["c_norm"],
#             posed_mask=pose_pack["posed_mask"]
#         )
#         def bbox_diag(x):
#             mn = x.min(0).values
#             mx = x.max(0).values
#             return (mx - mn).norm().item(), mn, mx

#         for j in range(num_samples):
#             gd, gmn, gmx = bbox_diag(gt_meshes[j]["vertices"].to(self.device))
#             rd0, _, _ = bbox_diag(reps_all[j].vertices)          # raw extractor output
#             rd1, _, _ = bbox_diag(reps_world[j].vertices)        # final
#             print(f"[j={j}] bbox diag: gt={gd:.3f}  raw_dec={rd0:.3f}  world_dec={rd1:.3f}")
#         self.renderer.rendering_options.bg_color = (0, 0, 0)
#         self.renderer.rendering_options.resolution = gt_images.shape[-1]

#         gt_render_results = self._render_batch(
#             [MeshExtractResult(vertices=m["vertices"].to(self.device), faces=m["faces"].to(self.device)) for m in gt_meshes],
#             exts, ints, return_types=["normal"]
#         )
#         ret_dict["gt_normal"] = {"value": self._flip_normal(gt_render_results["normal"], exts, ints), "type": "image"}

#         return_types = ["normal"]
#         if self.use_color:
#             return_types.append("color")

#         render_results = self._render_batch(reps_world, exts, ints, return_types=return_types)
#         ret_dict["rec_normal"] = {"value": render_results["normal"], "type": "image"}
#         if 'color' in return_types:
#             ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})
#         if 'normal_map' in return_types:
#             ret_dict.update({f'rec_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})

#         # render multiview
#         self.renderer.rendering_options.resolution = 512
#         ## Build camera
#         yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
#         yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
#         yaws = [y + yaws_offset for y in yaws]
#         pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

#         ## render each view
#         multiview_normals = []
#         multiview_normal_maps = []
#         miltiview_images = []
#         for yaw, pitch in zip(yaws, pitch):
#             orig = torch.tensor([
#                 np.sin(yaw) * np.cos(pitch),
#                 np.cos(yaw) * np.cos(pitch),
#                 np.sin(pitch),
#             ]).float().cuda() * 2
#             fov = torch.deg2rad(torch.tensor(30)).cuda()
#             extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
#             intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
#             extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
#             intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
#             render_results = self._render_batch(reps_world, extrinsics, intrinsics, return_types=return_types)
#             multiview_normals.append(render_results['normal'])
#             if 'color' in return_types:
#                 miltiview_images.append(render_results['color'])
#             if 'normal_map' in return_types:
#                 multiview_normal_maps.append(render_results['normal_map'])

#         ## Concatenate views
#         multiview_normals = torch.cat([
#             torch.cat(multiview_normals[:2], dim=-2),
#             torch.cat(multiview_normals[2:], dim=-2),
#         ], dim=-1)
#         ret_dict.update({f'multiview_normal': {'value': multiview_normals, 'type': 'image'}})
#         if 'color' in return_types:
#             miltiview_images = torch.cat([
#                 torch.cat(miltiview_images[:2], dim=-2),
#                 torch.cat(miltiview_images[2:], dim=-2),
#             ], dim=-1)
#             ret_dict.update({f'multiview_image': {'value': miltiview_images, 'type': 'image'}})
#         if 'normal_map' in return_types:
#             multiview_normal_maps = torch.cat([
#                 torch.cat(multiview_normal_maps[:2], dim=-2),
#                 torch.cat(multiview_normal_maps[2:], dim=-2),
#             ], dim=-1)
#             ret_dict.update({f'multiview_normal_map': {'value': multiview_normal_maps, 'type': 'image'}})
                            
#         return ret_dict



class SLatVaeMeshDecoderTrainerPose(BasicTrainer):
    """
    Trainer for structured latent VAE Mesh Decoder.
    
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
    """
    
    def __init__(
        self,
        *args,
        depth_loss_type: str = 'l1',
        lambda_depth: int = 1,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_tsdf: float = 0.01,
        lambda_color: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_loss_type = depth_loss_type
        self.lambda_depth = lambda_depth
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_tsdf = lambda_tsdf
        self.lambda_color = lambda_color
        self.use_color = self.lambda_color > 0
        
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                             "far" : 3}
        self.renderer = MeshRenderer(rendering_options, device=self.device)
        
    def _render_batch(self, reps: List[MeshExtractResult], extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                      return_types=['mask', 'normal', 'depth']) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
        Returns: 
            a dict with
                reg_loss : [N] tensor of regularization losses
                mask : [N x 1 x H x W] tensor of rendered masks
                normal : [N x 3 x H x W] tensor of rendered normals
                depth : [N x 1 x H x W] tensor of rendered depths
        """
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types)
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    @staticmethod
    def _tsdf_reg_loss(rep: MeshExtractResult, depth_map, extrinsics, intrinsics):
        with torch.no_grad():
            projected_pts, pts_depth = utils3d.torch.project_cv(
                extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v
            )
            projected_pts = (projected_pts - 0.5) * 2.0
            depth_map_res = depth_map.shape[1]
            gt_depth = torch.nn.functional.grid_sample(
                depth_map.reshape(1, 1, depth_map_res, depth_map_res),
                projected_pts.reshape(1, 1, -1, 2),
                mode='bilinear', padding_mode='border', align_corners=True
            )
            pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()

            # scale-aware truncation
            k = float(getattr(rep, "_k_world", 1.0))
            delta = (k / rep.res) * 3.0
            trunc_mask = pseudo_sdf > -delta

        gt_tsdf = pseudo_sdf[trunc_mask].clamp(-delta, delta)
        tsdf = rep.tsdf_s.flatten()[trunc_mask]
        return torch.mean((tsdf - gt_tsdf) ** 2)
    
    def _calc_tsdf_loss(self, reps : list[MeshExtractResult], depth_maps, extrinsics, intrinsics) -> torch.Tensor:
        tsdf_loss = 0.0
        for i, rep in enumerate(reps):
            tsdf_loss += self._tsdf_reg_loss(rep, depth_maps[i], extrinsics[i], intrinsics[i])
        return tsdf_loss / len(reps)
    
    @torch.no_grad()
    def _pose_to_world_params(self, R_row, s_aug, t_aug, c0, s_norm, c_norm, eps=1e-12):
        B = R_row.shape[0]
        s_aug  = s_aug.view(B, 1)
        s_norm = s_norm.view(B, 1)

        k = 1.0 / (s_aug.clamp_min(eps) * s_norm.clamp_min(eps))  # (B,1)
        tR = torch.einsum("bi,bij->bj", t_aug, R_row.transpose(1,2))  # (B,3)
        b = c_norm + (c0 / s_norm.clamp_min(eps)) - (tR * k)          # (B,3)
        return R_row, k, b


    
    
    def _transform_meshes_for_loss(
        self,
        reps_pose: list[MeshExtractResult],
        *,
        R_row, s_aug, t_aug, c0, s_norm, c_norm,
        posed_mask: torch.Tensor | None = None,   # [B] bool
        rotate_vertex_normal_attr: bool = False,  # only if you KNOW attrs contain normals
    ):
        B = len(reps_pose)

        if posed_mask is not None:
            posed_mask = posed_mask.to(device=R_row.device).bool()
            assert posed_mask.shape[0] == B

        R, k, b = self._pose_to_world_params(R_row, s_aug, t_aug, c0, s_norm, c_norm)
        # R: [B,3,3], k: [B,1], b: [B,3]

        reps_world = []
        for i, rep in enumerate(reps_pose):
            if posed_mask is not None and (not bool(posed_mask[i].item())):
                reps_world.append(rep)
                continue

            Ri = R[i]                  # [3,3]
            ki = k[i].view(1, 1)       # [1,1]
            bi = b[i].view(1, 3)       # [1,3]

            # --- vertices ---
            v_pose = rep.vertices      # [V,3]
            v_world = (v_pose @ Ri.t()) * ki + bi

            # --- vertex attrs (optional) ---
            attrs = rep.vertex_attrs
            if rotate_vertex_normal_attr and attrs is not None and attrs.shape[-1] >= 6:
                # Example convention: attrs[...,0:3]=color, attrs[...,3:6]=normal in [0,1]
                n01 = attrs[..., 3:6]
                n = n01 * 2.0 - 1.0
                n = (n @ Ri.t())  # rotate directions, NO scale/translation
                n = torch.nn.functional.normalize(n, dim=-1)
                attrs = torch.cat([attrs[..., 0:3], (n + 1.0) * 0.5, attrs[..., 6:]], dim=-1)

            # --- tsdf verts & values (IMPORTANT for tsdf loss) ---
            tsdf_v = rep.tsdf_v
            tsdf_s = rep.tsdf_s
            if tsdf_v is not None:
                tsdf_v = (tsdf_v @ Ri.t()) * ki + bi
            if tsdf_s is not None:
                # distances scale with geometry scaling
                tsdf_s = tsdf_s * ki.view(-1)

            # rebuild MeshExtractResult so face normals are consistent with transformed verts
            rep_w = MeshExtractResult(
                vertices=v_world,
                faces=rep.faces,
                vertex_attrs=attrs,
                res=rep.res
            )
            rep_w.reg_loss = rep.reg_loss
            rep_w.tsdf_v = tsdf_v
            rep_w.tsdf_s = tsdf_s

            # stash scale for tsdf truncation / smooth_l1 beta if you want it
            rep_w._k_world = ki.squeeze(0).squeeze(0)  # scalar tensor

            reps_world.append(rep_w)

        return reps_world



    @torch.no_grad()
    def _flip_normal(self, normal: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Flip normal to align with camera.
        """
        normal = normal * 2.0 - 1.0
        R = torch.zeros_like(extrinsics)
        R[:, :3, :3] = extrinsics[:, :3, :3]
        R[:, 3, 3] = 1.0
        view_dir = utils3d.torch.unproject_cv(
            utils3d.torch.image_uv(*normal.shape[-2:], device=self.device).reshape(1, -1, 2),
            torch.ones(*normal.shape[-2:], device=self.device).reshape(1, -1),
            R, intrinsics
        ).reshape(-1, *normal.shape[-2:], 3).permute(0, 3, 1, 2)
        unflip = (normal * view_dir).sum(1, keepdim=True) < 0
        normal *= unflip * 2.0 - 1.0
        return (normal + 1.0) / 2.0
    
    def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
        """
        Combination of L1, SSIM, and LPIPS loss.
        """
        if gt.shape[1] != 3:
            assert gt.shape[-1] == 3
            gt = gt.permute(0, 3, 1, 2)
        if pred.shape[1] != 3:
            assert pred.shape[-1] == 3
            pred = pred.permute(0, 3, 1, 2)
        terms = {
            f"{name}_loss" : l1_loss(gt, pred),
            f"{name}_loss_ssim" : 1 - ssim(gt, pred),
            f"{name}_loss_lpips" : lpips(gt, pred)
        }
        terms[f"{name}_loss_perceptual"] = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim + terms[f"{name}_loss_lpips"] * self.lambda_lpips
        return terms
    
    def geometry_losses(
        self,
        reps: List[MeshExtractResult],
        mesh: List[Dict],
        normal_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        with torch.no_grad():
            gt_meshes = []
            for i in range(len(reps)):
                gt_mesh = MeshExtractResult(mesh[i]['vertices'].to(self.device), mesh[i]['faces'].to(self.device))
                gt_meshes.append(gt_mesh)
            target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=['mask', 'depth', 'normal'])
            target['normal'] = self._flip_normal(target['normal'], extrinsics, intrinsics)
                
        terms = edict(geo_loss = 0.0)
        if self.lambda_tsdf > 0:
            tsdf_loss = self._calc_tsdf_loss(reps, target['depth'], extrinsics, intrinsics)
            terms['tsdf_loss'] = tsdf_loss
            terms['geo_loss'] += tsdf_loss * self.lambda_tsdf
        
        return_types = ['mask', 'depth', 'normal', 'normal_map'] if self.use_color else ['mask', 'depth', 'normal']
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
        
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            for k, v in buffer.items():
                buffer[k] = v[success_mask]
            for k, v in target.items():
                target[k] = v[success_mask]
            
            terms['mask_loss'] = l1_loss(buffer['mask'], target['mask']) 
            if self.depth_loss_type == 'l1':
                terms['depth_loss'] = l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'])
            elif self.depth_loss_type == 'smooth_l1':
                terms['depth_loss'] = smooth_l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'], beta=1.0 / (2 * reps[0].res))
            else:
                raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
            terms.update(self._perceptual_loss(buffer['normal'] * target['mask'], target['normal'] * target['mask'], 'normal'))
            terms['geo_loss'] = terms['geo_loss'] + terms['mask_loss'] + terms['depth_loss'] * self.lambda_depth + terms['normal_loss_perceptual']
            if self.use_color and normal_map is not None:
                terms.update(self._perceptual_loss(normal_map[success_mask], buffer['normal_map'], 'normal_map'))
                terms['geo_loss'] = terms['geo_loss'] + terms['normal_map_loss_perceptual'] * self.lambda_color
                
        return terms
      
    def color_losses(self, reps, image, alpha, extrinsics, intrinsics):
        terms = edict(color_loss = torch.tensor(0.0, device=self.device))
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=['color'])
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            terms.update(self._perceptual_loss((image * alpha[:, None])[success_mask], buffer['color'][success_mask], 'color'))
            terms['color_loss'] = terms['color_loss'] + terms['color_loss_perceptual'] * self.lambda_color
        return terms
    
    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        mesh: List[Dict],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        # --- pose params ---
        R_row: torch.Tensor,
        s_aug: torch.Tensor,
        t_aug: torch.Tensor,
        c0: torch.Tensor,
        s_norm: torch.Tensor,
        c_norm: torch.Tensor,
        origin: torch.Tensor,
        voxel_size: torch.Tensor,
        posed_mask: torch.Tensor = None,
        normal_map: torch.Tensor = None,
    ):
        reps_pose = self.training_models['decoder'](latents)

        reps_world = self._transform_meshes_for_loss(
            reps_pose,
            R_row=R_row, s_aug=s_aug, t_aug=t_aug, c0=c0, s_norm=s_norm, c_norm=c_norm,
            posed_mask=posed_mask,
            rotate_vertex_normal_attr=False,  # set True only if your attrs really contain normals
        )

        self.renderer.rendering_options.resolution = image.shape[-1]

        terms = edict(loss=0.0)

        # reg loss (same scalar, but use world list for consistency)
        terms['reg_loss'] = sum([rep.reg_loss for rep in reps_world]) / len(reps_world)
        terms['loss'] = terms['loss'] + terms['reg_loss']

        geo_terms = self.geometry_losses(reps_world, mesh, normal_map, extrinsics, intrinsics)
        terms.update(geo_terms)
        terms['loss'] = terms['loss'] + terms['geo_loss']

        if self.use_color:
            color_terms = self.color_losses(reps_world, image, alpha, extrinsics, intrinsics)
            terms.update(color_terms)
            terms['loss'] = terms['loss'] + terms['color_loss']

        return terms, {}

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

        # -------------------------
        # small recursive mover (handles torch tensors + SparseTensor + lists/dicts)
        # -------------------------
        def _to_device(x, device):
            try:
                import trellis.modules.sparse as sp
                is_sparse = isinstance(x, sp.SparseTensor)
            except Exception:
                is_sparse = False

            if is_sparse:
                # sp.SparseTensor typically supports .to/.cuda
                return x.to(device) if hasattr(x, "to") else x.cuda()
            if torch.is_tensor(x):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _to_device(v, device) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                y = [_to_device(v, device) for v in x]
                return type(x)(y) if not isinstance(x, tuple) else tuple(y)
            return x

        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        # -------------------------
        # collect a small fixed number of samples
        # -------------------------
        num_samples = min(int(num_samples), 4)  # keep snapshots small like your other trainers

        ret_dict = {}
        gt_images = []
        gt_normal_maps = []
        gt_meshes = []
        exts = []
        ints = []
        reps_pose = []

        # pose metadata (must be present in your pose-latent dataset)
        pose_pack = {
            "R_row": [],
            "s_aug": [],
            "t_aug": [],
            "c0": [],
            "s_norm": [],
            "c_norm": [],
            # optional
            "origin": [],
            "voxel_size": [],
            "posed_mask": [],
        }

        it = iter(dataloader)
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(it)
            args = _to_device(data, self.device)

            # GT image for display
            gt_images.append(args["image"][:batch] * args["alpha"][:batch, None])

            if self.use_color and ("normal_map" in args):
                gt_normal_maps.append(args["normal_map"][:batch])

            # GT meshes (keep dicts; vertices/faces should already be tensors)
            gt_meshes.extend(args["mesh"][:batch])

            # WORLD cameras (vanilla)
            exts.append(args["extrinsics"][:batch])
            ints.append(args["intrinsics"][:batch])

            # pose params (aligned with this batch)
            for k in pose_pack.keys():
                if k in args and args[k] is not None:
                    pose_pack[k].append(args[k][:batch])
                else:
                    pose_pack[k].append(None)

            # decode POSE-space meshes from stored pose-latents
            reps_pose.extend(self.models["decoder"](args["latents"][:batch] if hasattr(args["latents"], "__getitem__") else args["latents"]))

        # concat GT display tensors
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict["gt_image"] = {"value": gt_images, "type": "image"}
        if self.use_color and len(gt_normal_maps) > 0:
            gt_normal_maps = torch.cat(gt_normal_maps, dim=0)
            ret_dict["gt_normal_map"] = {"value": gt_normal_maps, "type": "image"}

        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)

        # -------------------------
        # concat pose args (skip optional all-None fields)
        # -------------------------
        pose_args = {}
        for k, chunks in pose_pack.items():
            if all(c is None for c in chunks):
                pose_args[k] = None
            else:
                assert not any(c is None for c in chunks), f"Missing pose field {k} for some batches"
                pose_args[k] = torch.cat(chunks, dim=0)

        # REQUIRED pose fields
        for k in ["R_row", "s_aug", "t_aug", "c0", "s_norm", "c_norm"]:
            assert pose_args.get(k, None) is not None, f"Dataset must provide '{k}' for pose snapshot."

        # -------------------------
        # POSE -> WORLD meshes (critical)
        #   expects you implemented _transform_meshes_for_loss + _pose_to_world_params
        # -------------------------
        reps_world = self._transform_meshes_for_loss(
            reps_pose,
            R_row=pose_args["R_row"],
            s_aug=pose_args["s_aug"],
            t_aug=pose_args["t_aug"],
            c0=pose_args["c0"],
            s_norm=pose_args["s_norm"],
            c_norm=pose_args["c_norm"],
            posed_mask=pose_args.get("posed_mask", None),
        )

        # -------------------------
        # render single view (GT cameras, WORLD frame)
        # -------------------------
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        if hasattr(self.renderer.rendering_options, "bg_color"):
            self.renderer.rendering_options.bg_color = (0, 0, 0)

        # GT normal render (WORLD meshes)
        gt_render_results = self._render_batch(
            [
                MeshExtractResult(
                    vertices=m["vertices"].to(self.device),
                    faces=m["faces"].to(self.device),
                )
                for m in gt_meshes
            ],
            exts, ints,
            return_types=["normal"],
        )
        ret_dict["gt_normal"] = {
            "value": self._flip_normal(gt_render_results["normal"], exts, ints),
            "type": "image",
        }

        # REC render
        return_types = ["normal"]
        if self.use_color:
            return_types.append("color")
            if len(gt_normal_maps) > 0:
                return_types.append("normal_map")

        render_results = self._render_batch(reps_world, exts, ints, return_types=return_types)
        ret_dict["rec_normal"] = {"value": render_results["normal"], "type": "image"}
        if "color" in render_results:
            ret_dict["rec_image"] = {"value": render_results["color"], "type": "image"}
        if "normal_map" in render_results:
            ret_dict["rec_normal_map"] = {"value": render_results["normal_map"], "type": "image"}

        # -------------------------
        # render multiview (synthetic cameras, WORLD frame)
        # -------------------------
        self.renderer.rendering_options.resolution = 512

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        multiview_normals = []
        multiview_images = []
        multiview_normal_maps = []

        for yaw, pit in zip(yaws, pitch):
            orig = torch.tensor(
                [np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), np.sin(pit)],
                device=self.device,
                dtype=torch.float32,
            ) * 2.0

            fov = torch.deg2rad(torch.tensor(30.0, device=self.device))
            extr = utils3d.torch.extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
                torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32),
            )
            intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

            extr = extr.unsqueeze(0).expand(num_samples, -1, -1)
            intr = intr.unsqueeze(0).expand(num_samples, -1, -1)

            mv = self._render_batch(reps_world, extr, intr, return_types=return_types)
            multiview_normals.append(mv["normal"])
            if "color" in mv:
                multiview_images.append(mv["color"])
            if "normal_map" in mv:
                multiview_normal_maps.append(mv["normal_map"])

        # concat 2x2 grid
        mv_norm = torch.cat(
            [torch.cat(multiview_normals[:2], dim=-2), torch.cat(multiview_normals[2:], dim=-2)],
            dim=-1,
        )
        ret_dict["multiview_normal"] = {"value": mv_norm, "type": "image"}

        if len(multiview_images) > 0:
            mv_img = torch.cat(
                [torch.cat(multiview_images[:2], dim=-2), torch.cat(multiview_images[2:], dim=-2)],
                dim=-1,
            )
            ret_dict["multiview_image"] = {"value": mv_img, "type": "image"}

        if len(multiview_normal_maps) > 0:
            mv_nm = torch.cat(
                [torch.cat(multiview_normal_maps[:2], dim=-2), torch.cat(multiview_normal_maps[2:], dim=-2)],
                dim=-1,
            )
            ret_dict["multiview_normal_map"] = {"value": mv_nm, "type": "image"}

        return ret_dict
