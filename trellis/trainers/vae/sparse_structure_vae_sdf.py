from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
import os

class EikonalLoss(torch.nn.Module):
    def __init__(self, spacing=1.0):
        super().__init__()
        # Create the kernel ONCE during initialization.
        kernel_dx = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[-1,0,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[0,0,0],[0,-1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]]], dtype=torch.float32)
        kernel_dz = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[0,0,0],[0,1,0]],[[0,0,0],[0,0,0],[0,0,0]]]], dtype=torch.float32)
        self.spacing = spacing
        # Stack to a single [3, 1, 3, 3, 3] kernel
        combined_kernel = torch.stack([kernel_dx, kernel_dy, kernel_dz], dim=0)
        # Register the kernel as a buffer. This makes it part of the module's state
        # and handles device placement automatically.
        self.register_buffer('kernel', combined_kernel)
        
 
    def forward(self, sdf_grid):
        # The kernel is now accessed via self.kernel. It will always be on the
        # same device as the module, which should be the same as the sdf_grid.
        # No allocation or transfer happens here.
        all_gradients = F.conv3d(sdf_grid, self.kernel, padding=1) / (2.0*self.spacing)
        gradient_norm = torch.norm(all_gradients, p=2, dim=1, keepdim=True)
        return ((gradient_norm - 1.0) ** 2).mean()


class CombinedGeometricLoss(torch.nn.Module):
    """
    Computes a combined Eikonal and Normal Consistency loss for a grid-based SDF.
    This module incorporates several optimizations:
    1.  The SDF gradient is computed only once and reused.
    2.  The convolution kernel is registered as a buffer to avoid reallocation.
    3.  Voxel spacing is properly accounted for.
    4.  Optional stochastic subsampling is supported for faster training.
    """
    def __init__(self, spacing=1.0, num_samples=None):
        super().__init__()
        self.spacing = spacing
        self.num_samples = num_samples
        # Create the kernel ONCE and register it as a buffer.
        kernel_dx = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[-1,0,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[0,0,0],[0,-1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]]], dtype=torch.float32)
        kernel_dz = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[0,0,0],[0,1,0]],[[0,0,0],[0,0,0],[0,0,0]]]], dtype=torch.float32)
        combined_kernel = torch.stack([kernel_dx, kernel_dy, kernel_dz], dim=0)
        self.register_buffer('kernel', combined_kernel)
        self.kernel.to('cuda')

    def forward(self, s_pred_grid, s_gt_grid):
        # --- 1. Reusable Gradient Calculation ---
        # The denominator in the central difference formula is (2 * h)
        denominator = 2.0 * self.spacing
        with torch.amp.autocast('cuda', enabled=False):
            # Calculate gradients for both predicted and ground truth grids
            pred_gradients = F.conv3d(s_pred_grid, self.kernel, padding=1) / denominator
            gt_gradients = F.conv3d(s_gt_grid, self.kernel, padding=1) / denominator

            # --- 2. Eikonal Loss Calculation ---
            pred_gradient_norm = torch.norm(pred_gradients, p=2, dim=1) # Norm is now [B, D, H, W]
            # Add clamp
            pred_gradient_norm = torch.clamp(pred_gradient_norm, min=1e-4, max=10.0)
            eikonal_per_voxel_loss = (pred_gradient_norm - 1.0) ** 2
            # --- 3. Normal Consistency Loss Calculation ---
            # Create a mask for the narrow band around the surface
            surface_mask = torch.abs(s_gt_grid.squeeze(1)) < (2 * self.spacing) # Mask is [B, D, H, W]
            # Add clamp
            gt_grad_norm = torch.norm(gt_gradients, p=2, dim=1, keepdim=True)
            gt_grad_norm = torch.clamp(gt_grad_norm, min=1e-4, max=10.0)
            # Normalize the gradient vectors to get normals. Add epsilon for stability.
            # Note: We reuse pred_gradient_norm here.
            pred_normals = pred_gradients / (pred_gradient_norm.unsqueeze(1) + 1e-8)
            gt_normals = gt_gradients /gt_grad_norm #(torch.norm(gt_gradients, p=2, dim=1, keepdim=True) + 1e-8)
            # Cosine similarity is the dot product of the unit vectors
            # We need to sum along the channel dimension (dim=1)
            cos_sim = torch.sum(pred_normals * gt_normals, dim=1) # Result is [B, D, H, W]
            # The per-voxel loss before taking the mean
            normal_per_voxel_loss = 1.0 - cos_sim
            
            # --- 4. Optional Stochastic Subsampling ---
            if self.num_samples is not None and self.num_samples < s_pred_grid.numel():
                # Generate random indices once and reuse for all losses
                num_voxels = s_pred_grid.numel()
                indices = torch.randint(0, num_voxels, (self.num_samples,), device=s_pred_grid.device)
                # Sample the per-voxel losses using the same indices
                eikonal_loss = eikonal_per_voxel_loss.view(-1)[indices].mean()
    
                # For normal loss, we only sample from the surface region for efficiency
                masked_normal_loss = torch.masked_select(normal_per_voxel_loss, surface_mask)
                if masked_normal_loss.numel() > 0:
                    # To keep it simple, if the number of surface points is large enough, we sample from it.
                    # A more complex strategy could be used if needed.
                    num_surface_samples = min(self.num_samples, masked_normal_loss.numel())
                    surf_indices = torch.randint(0, masked_normal_loss.numel(), (num_surface_samples,), device=s_pred_grid.device)
                    normal_loss = masked_normal_loss[surf_indices].mean()
                else:
                    normal_loss = torch.tensor(0.0, device=s_pred_grid.device) # No surface points found
            else:
                # --- 5. Full Grid Loss Calculation ---
                eikonal_loss = eikonal_per_voxel_loss.mean()
                normal_loss = torch.masked_select(normal_per_voxel_loss, surface_mask).mean()

        if torch.isnan(pred_gradients).any():
            print("⚠️ NaN in pred_gradients")
        if torch.isnan(pred_gradient_norm).any():
            print("⚠️ NaN in pred_gradient_norm")
        if torch.isnan(eikonal_per_voxel_loss).any():
            print("⚠️ NaN in eikonal loss")
        if torch.isnan(normal_per_voxel_loss).any():
            print("⚠️ NaN in normal loss")
        print(f"grad_norm range: [{pred_gradient_norm.min().item():.3f}, {pred_gradient_norm.max().item():.3f}]")
        print(f"logits range: [{s_pred_grid.min().item():.3f}, {s_pred_grid.max().item():.3f}]")
        # logits_range = (s_pred_grid.min().item(), s_pred_grid.max().item())
        return eikonal_loss, normal_loss
    
class SparseStructureVaeSDFTrainer(BasicTrainer):
    """
    Trainer for Sparse Structure VAE.
    
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
        
        loss_type (str): Loss type. 'bce' for binary cross entropy, 'l1' for L1 loss, 'dice' for Dice loss.
        lambda_kl (float): KL divergence loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type='bce',
        lambda_kl=1e-6,
        lambda_eikonal=0.1,
        lambda_normal=0.1,  # Normal consistency loss weight
        lambda_l1=10.0,  # L1 loss weight
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.lambda_l1 = lambda_l1  # L1 loss weight
        self.lambda_eikonal = lambda_eikonal
        self.lambda_normal = lambda_normal  # Normal consistency loss weight
        resolution = 64
        self.eikonal_loss = EikonalLoss(spacing=2/resolution).to('cuda')
        self.combined_geometric_loss = CombinedGeometricLoss(spacing=2/resolution).to('cuda')
    
    def training_losses(
        self,
        ss: torch.Tensor,
        sdf: torch.Tensor,
        instance: str,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # self.training_models['encoder'].eval()
        # with torch.no_grad():
        for name, p in self.training_models['encoder'].named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                with open(os.path.join(self.output_dir,"nan_debug.log"), "a") as f:
                    f.write(f"Parameter {name} of encoder has NaNs or Infs! Step {self.step}")
        for name, p in self.training_models['decoder'].named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                with open(os.path.join(self.output_dir,"nan_debug.log"), "a") as f:
                    f.write(f"Parameter {name} of decoder has NaNs or Infs! Step {self.step}")
        z, mean, logvar = self.training_models['encoder'](sdf.float(), sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)
        if not torch.isfinite(logits).all():
            print(f"NaNs in logits before loss. Aborting step {self.step}.")
            return {"loss": torch.tensor(float('nan'), device=logits.device)}, {}
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        terms = edict(loss = 0.0)
        if self.loss_type == 'bce':
            terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["bce"]
        elif self.loss_type == 'eikonal':
            sdf = sdf.unsqueeze(1)  # Ensure sdf is [N, 1, H, W, D]
            terms["l1"] = F.l1_loss(logits, sdf, reduction='mean')
            terms["eikonal"] = self.eikonal_loss(logits)
            terms["loss"] = terms["loss"] + terms["l1"] + terms["eikonal"] * self.lambda_eikonal
        elif self.loss_type == 'geometric':
            # sdf = sdf.unsqueeze(1)  # Ensure sdf is [N, 1, H, W, D]
            terms["l1"] = F.l1_loss(logits, sdf, reduction='mean') * self.lambda_l1
            terms["eikonal"], terms['normal']= self.combined_geometric_loss(logits, sdf)
            terms["eikonal"] = terms["eikonal"] * self.lambda_eikonal
            terms['normal'] = terms['normal']*self.lambda_normal
            terms["loss"] = terms["loss"] + terms["l1"] +  terms["eikonal"] + terms['normal']
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            terms["loss"] = terms["loss"] + terms["dice"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        # print(f"[Step {self.step}] NaN detected! Instance: {' '.join(instance)}\n")
        if not torch.isfinite(terms["loss"]):
            with open(os.path.join(self.output_dir,"nan_debug.log", "a") as f:
                f.write(f"[Step {self.step}] NaN detected! Instance: {' '.join(instance)}\n")
                f.write(f"  l1: {terms['l1']}, eikonal: {terms['eikonal']}, normal: {terms.get('normal', 'n/a')}\n")
                f.write(f"  logits max: {logits.max().item()}, min: {logits.min().item()}\n")
                f.write(f"logvar stats: {logvar.min().item()}, {logvar.max().item()}")
                f.write(f"mean stats: {mean.min().item()}, {mean.max().item()}")
        return terms, {"logvar_min": logvar.min().item(), "logvar_min": logvar.min().item(),"mean_min": mean.min().item(), "mean_min": mean.min().item(),}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
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
        gts = []
        recons = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            z = self.models['encoder'](args['sdf'].float(), sample_posterior=False)
            logits = self.models['decoder'](z)
            #recon = (logits <= 0).long()
            gts.append(args['sdf'])
            recons.append(logits)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'},
        }
        return sample_dict
