from typing import Dict, Optional, Tuple
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from easydict import EasyDict as edict

from ... import models
from ...utils.general_utils import dict_reduce
from .flow_matching import ImageConditionedFlowMatchingCFGTrainerConditioned


class _ConditionCopyWrapper(torch.nn.Module):
    """
    Forward wrapper that prevents SparseStructureFlowModelConditioned from
    mutating the condition dictionary reused across Euler sampling steps.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _copy_cond(cond):
        if isinstance(cond, dict):
            return dict(cond)
        return cond

    def forward(self, x, t, cond=None):
        return self.model(x, t, self._copy_cond(cond))


class ImageConditionedFlowMatchingCFGDistillationTrainerConditioned(
    ImageConditionedFlowMatchingCFGTrainerConditioned
):
    """
    Distill an image-conditioned sparse-structure flow model.

    The student is the normal ``denoiser`` passed through the config. The teacher
    is loaded from a separate TRELLIS config and checkpoint, kept frozen, and used
    as an additional velocity target.
    """

    def __init__(
        self,
        *args,
        teacher_config_path: str,
        teacher_ckpt_path: str,
        teacher_model_key: str = "denoiser",
        teacher_strict_load: bool = True,
        distill_loss_weight: float = 1.0,
        target_loss_weight: float = 0.25,
        use_physics_losses: bool = True,
        use_visual_losses: bool = False,
        lambda_presence: float = 0.0,
        lambda_carving: float = 0.0,
        visual_min_mask_px: int = 30,
        visual_presence_margin: float = 0.015625,
        **kwargs,
    ):
        self.teacher_config_path = teacher_config_path
        self.teacher_ckpt_path = teacher_ckpt_path
        self.teacher_model_key = teacher_model_key
        self.teacher_strict_load = teacher_strict_load
        self.distill_loss_weight = distill_loss_weight
        self.target_loss_weight = target_loss_weight
        self.use_physics_losses = use_physics_losses
        self.use_visual_losses = use_visual_losses
        self.lambda_presence = lambda_presence
        self.lambda_carving = lambda_carving
        self.visual_min_mask_px = visual_min_mask_px
        self.visual_presence_margin = visual_presence_margin
        self.teacher_denoiser = None

        super().__init__(*args, **kwargs)

    def init_models_and_more(self, **kwargs):
        super().init_models_and_more(**kwargs)
        self.teacher_denoiser = self._load_teacher()

    def _load_teacher_state_dict(self, ckpt_path: str) -> Dict[str, torch.Tensor]:
        if not ckpt_path:
            raise ValueError("teacher_ckpt_path must be provided for distillation.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(ckpt_path, device="cpu")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        return ckpt

    def _load_teacher(self) -> torch.nn.Module:
        if not self.teacher_config_path:
            raise ValueError("teacher_config_path must be provided for distillation.")
        if not os.path.exists(self.teacher_config_path):
            raise FileNotFoundError(f"Teacher config not found: {self.teacher_config_path}")

        with open(self.teacher_config_path, "r") as fp:
            teacher_cfg = json.load(fp)

        teacher_spec = teacher_cfg["models"][self.teacher_model_key]
        teacher_args = dict(teacher_spec["args"])
        # The teacher never needs gradients; activation checkpointing would only
        # re-run its blocks for nothing (use_checkpoint is not gated on .training).
        teacher_args["use_checkpoint"] = False
        teacher = getattr(models, teacher_spec["name"])(**teacher_args)
        state_dict = self._load_teacher_state_dict(self.teacher_ckpt_path)
        missing, unexpected = teacher.load_state_dict(
            state_dict,
            strict=self.teacher_strict_load,
        )

        teacher = teacher.cuda().eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        if self.is_master:
            print("\nLoaded distillation teacher:")
            print(f"  config: {self.teacher_config_path}")
            print(f"  ckpt:   {self.teacher_ckpt_path}")
            if missing:
                print(f"  missing keys: {len(missing)}")
            if unexpected:
                print(f"  unexpected keys: {len(unexpected)}")

        return teacher

    @staticmethod
    def _copy_cond(cond):
        if isinstance(cond, dict):
            return dict(cond)
        return cond

    def _add_physics_losses(
        self,
        terms: edict,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
        x0_hand: torch.Tensor,
        touch: Optional[torch.Tensor],
        kept: torch.Tensor,
    ) -> edict:
        # Faithful port of the FIXED physics block in
        # ImageConditionedFlowMatchingCFGTrainerConditioned.training_losses
        # (flow_matching.py). The previous version here predated the stage-2
        # corrections: it reconstructed x0 as (1-sigma)*noise - pred (not an
        # estimator of x_0 -- it carries the full v-error and uses ground-truth
        # noise), had no CFG-drop gating, normalised the time weighting by w.sum()
        # (scale-invariant, so an all-high-t batch still got a full-magnitude
        # physics loss), had no margin, masked the hand at sdf<0 (so NI and
        # contact fought over the contact shell), and had no GT-latent floors.
        lambda_ni = self.get_lambda_ni() if self.lambda_ni_max is not None else 0.0
        lambda_contact = self.lambda_contact or 0.0
        physics_on = (self.use_physics_losses and self.use_encoding_hand
                      and (lambda_ni > 0 or lambda_contact > 0))
        if not physics_on:
            return terms

        B = x_0.shape[0]
        p = self.physics_time_power
        w = (1.0 - t).clamp(0.0, 1.0).pow(p) * kept                     # [B]

        x0_pred, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred)

        sdf_obj = self.ss_dec(x0_pred).float()                          # [B,1,64,64,64]
        with torch.no_grad():
            sdf_hand = self.ss_dec(x0_hand).float()
            sdf_gt = (self.ss_dec(x_0).float()
                      if (self.contact_relative or self.ni_relative) else None)

        max_pen = 0.1
        margin = self.ni_margin
        obj_inside = max_pen * torch.tanh(F.relu(margin - sdf_obj) / max_pen)
        hand_mask = (sdf_hand < -margin).float()

        num = (obj_inside * hand_mask).view(B, -1).sum(dim=1)
        den = hand_mask.view(B, -1).sum(dim=1).clamp_min(1.0)
        ni_per_sample = num / den

        if self.ni_relative:
            gt_inside = max_pen * torch.tanh(F.relu(margin - sdf_gt) / max_pen)
            ni_floor = (gt_inside * hand_mask).view(B, -1).sum(dim=1) / den
            terms["ni_floor"] = ni_floor.mean()
            ni_per_sample = F.relu(ni_per_sample - ni_floor)

        ni_loss_raw = (w * ni_per_sample).mean()
        terms["ni_loss_raw"] = ni_loss_raw
        terms["ni_loss"] = lambda_ni * ni_loss_raw
        terms["loss"] = terms["loss"] + lambda_ni * ni_loss_raw

        if self.use_touch and (touch is not None) and lambda_contact > 0:
            if self.contact_mode == 'soft':
                contact_w = torch.exp(-touch[:, 1].float() / max(self.contact_sigma, 1e-6))
            else:
                contact_w = touch[:, 0].float()

            sdf_abs = sdf_obj.abs().squeeze(1)
            num = (contact_w * sdf_abs).view(B, -1).sum(dim=1)
            den = contact_w.view(B, -1).sum(dim=1).clamp_min(1.0)
            contact_per_sample = num / den

            if self.contact_relative:
                contact_floor = ((contact_w * sdf_gt.abs().squeeze(1))
                                 .view(B, -1).sum(dim=1) / den)
                terms["contact_floor"] = contact_floor.mean()
                contact_per_sample = F.relu(contact_per_sample - contact_floor)

            contact_raw = (w * contact_per_sample).mean()
            terms["contact_raw"] = contact_raw
            terms["contact_loss"] = contact_raw * lambda_contact
            terms["loss"] = terms["loss"] + contact_raw * lambda_contact

        return terms

    def _add_visual_losses(self, terms, pred, x_t, t, x_0, mask_obj, mask_hand, kept):
        """Silhouette losses (user proposal 2026-08-26; validated in
        tools/validate_mask_column_correspondence.py). Grid view axis is -z, so
        each mask pixel corresponds to a voxel column along z after the
        orientation map found by validation (mask_grid[x,y] = mask[63-y, x]).
        Per-sample 2-parameter calibration (area-ratio scale, centroid shift)
        against the GT-SDF z-projection absorbs the collage's aspect-fit and
        perspective wobble (presence violations 10.5% -> 2.0% measured).
        Presence: ERODED calibrated object mask (minus hand) -> soft-min SDF
        along the column must be < 0. Carving: outside the DILATED
        (object OR hand) mask the column is provably empty -> penalize sdf < 0.
        Views with a tiny visible object (< visual_min_mask_px at 64^2) get
        zero weight - their silhouette carries no signal."""
        B = x_0.shape[0]
        w = (1.0 - t).clamp(0.0, 1.0).pow(self.physics_time_power) * kept   # [B]

        x0_pred, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred)
        sdf_obj = self.ss_dec(x0_pred).float().squeeze(1)                    # [B,64,64,64] (x,y,z)
        with torch.no_grad():
            sdf_gt = self.ss_dec(x_0).float().squeeze(1)
            proj_gt = (sdf_gt < 0).any(dim=3).float()                        # [B,64,64] (x,y)

            def to_grid(m):                                                  # [B,H,W] image -> [B,64,64] grid (x,y)
                m = F.interpolate(m.unsqueeze(1).float(), size=(64, 64),
                                  mode="bilinear", align_corners=False)
                return m.transpose(2, 3).flip(3).squeeze(1)

            m_obj, m_hand = to_grid(mask_obj), to_grid(mask_hand)

            # per-sample similarity calibration: mask -> GT projection
            xs = torch.linspace(-1, 1, 64, device=m_obj.device)
            gy, gx = torch.meshgrid(xs, xs, indexing="ij")
            def moments(m):
                a = m.sum(dim=(1, 2)).clamp_min(1e-6)
                cx = (m * gy[None]).sum(dim=(1, 2)) / a                      # dim0 coord
                cy = (m * gx[None]).sum(dim=(1, 2)) / a                      # dim1 coord
                return a, cx, cy
            a_p, cx_p, cy_p = moments(proj_gt)
            a_m, cx_m, cy_m = moments((m_obj > 0.5).float())
            s_cal = (a_p / a_m).sqrt().clamp(0.7, 1.4)
            # affine_grid theta maps OUTPUT coords -> INPUT sampling coords:
            # sample mask at (out - c_p)/s + c_m
            theta = torch.zeros(B, 2, 3, device=m_obj.device)
            theta[:, 0, 0] = 1.0 / s_cal
            theta[:, 1, 1] = 1.0 / s_cal
            theta[:, 0, 2] = cy_m - cy_p / s_cal
            theta[:, 1, 2] = cx_m - cx_p / s_cal
            grid = F.affine_grid(theta, (B, 1, 64, 64), align_corners=False)
            def cal(m):
                return F.grid_sample(m.unsqueeze(1), grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=False).squeeze(1)
            m_obj_c, m_hand_c = cal(m_obj), cal(m_hand)

            pool = lambda m, k: F.max_pool2d(m.unsqueeze(1), k, 1, k // 2).squeeze(1)
            m_obj_er = 1.0 - pool(1.0 - m_obj_c, 3)                          # erode 1px
            union_dil = pool(torch.maximum(m_obj_c, m_hand_c), 5)            # dilate 2px
            presence_px = ((m_obj_er > 0.5) & (m_hand_c < 0.3)).float()      # [B,64,64]
            empty_px = (union_dil < 0.2).float()
            valid = (((m_obj > 0.5).float().sum(dim=(1, 2)) >= self.visual_min_mask_px)
                     .float())                                               # [B]

        tau = 50.0
        softmin = -torch.logsumexp(-sdf_obj * tau, dim=3) / tau              # [B,64,64]
        pres_ps = (F.relu(softmin + self.visual_presence_margin) * presence_px
                   ).sum(dim=(1, 2)) / presence_px.sum(dim=(1, 2)).clamp_min(1.0)
        carv_ps = (F.relu(-sdf_obj) * empty_px.unsqueeze(3)
                   ).sum(dim=(1, 2, 3)) / (empty_px.sum(dim=(1, 2)) * 64).clamp_min(1.0)

        wv = w * valid
        presence_raw = (wv * pres_ps).mean()
        carving_raw = (wv * carv_ps).mean()
        terms["presence_raw"] = presence_raw
        terms["carving_raw"] = carving_raw
        terms["presence_loss"] = self.lambda_presence * presence_raw
        terms["carving_loss"] = self.lambda_carving * carving_raw
        terms["visual_valid_frac"] = valid.mean()
        terms["loss"] = (terms["loss"] + self.lambda_presence * presence_raw
                         + self.lambda_carving * carving_raw)
        return terms

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        num_samples = 8
        if self.snapshot_indices is None:
            g = torch.Generator().manual_seed(1234)
            self.snapshot_indices = torch.randperm(len(self.dataset), generator=g)[:num_samples]

        indices = self.snapshot_indices[:num_samples]

        base_dataset = copy.deepcopy(self.dataset)
        base_dataset.inference = True
        snapshot_dataset = Subset(base_dataset, indices)

        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

        sampler = self.get_sampler()
        student_model = _ConditionCopyWrapper(self.models["denoiser"])
        teacher_model = _ConditionCopyWrapper(self.teacher_denoiser)

        sample_gt = []
        sample_student = []
        sample_teacher = []
        sample_hand = []
        cond_vis = []

        loader_iter = iter(dataloader)
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(loader_iter)
            data = {
                k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch]
                for k, v in data.items()
            }

            noise = torch.randn_like(data["x_0"])
            sample_gt.append(data["x_0"])
            sample_hand.append(data["x0_hand"])
            cond_vis.append(self.vis_cond(**data))

            del data["x_0"]
            args = self.get_inference_cond(**data)

            student_res = sampler.sample(
                student_model,
                noise=noise.clone(),
                **args,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )
            teacher_res = sampler.sample(
                teacher_model,
                noise=noise.clone(),
                **args,
                steps=50,
                cfg_strength=3.0,
                verbose=verbose,
            )

            sample_student.append(student_res.samples)
            sample_teacher.append(teacher_res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample_student = torch.cat(sample_student, dim=0)
        sample_teacher = torch.cat(sample_teacher, dim=0)
        sample_hand = torch.cat(sample_hand, dim=0)

        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample_student, "type": "sample"},
            "sample_teacher": {"value": sample_teacher, "type": "sample"},
            "sample_hand": {"value": sample_hand, "type": "sample"},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            "value": lambda x: torch.cat(x, dim=0),
            "type": lambda x: x[0],
        }))

        return sample_dict

    def training_losses(
        self,
        x_0: torch.Tensor,
        x0_hand: torch.Tensor,
        cond=None,
        mask_hand=None,
        mask_obj=None,
        cond_mask=None,
        touch=None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        cond_dict = self.get_cond(cond, mask_hand, mask_obj, cond_mask, x0_hand, touch, **kwargs)

        with torch.no_grad():
            teacher_pred = self.teacher_denoiser(
                x_t,
                t * 1000,
                self._copy_cond(cond_dict),
                **kwargs,
            )

        pred = self.training_models["denoiser"](
            x_t,
            t * 1000,
            self._copy_cond(cond_dict),
            **kwargs,
        )

        assert pred.shape == teacher_pred.shape == noise.shape == x_0.shape

        target = self.get_v(x_0, noise, t)
        target_mse = F.mse_loss(pred, target)
        distill_mse = F.mse_loss(pred, teacher_pred.detach())

        terms = edict()
        terms["target_mse"] = target_mse
        terms["distill_mse"] = distill_mse
        terms["mse"] = target_mse
        terms["loss"] = (
            self.target_loss_weight * target_mse
            + self.distill_loss_weight * distill_mse
        )

        # Physics must be silent on samples whose hand conditioning was CFG-dropped
        # (same gating as the teacher's trainer).
        B = x_0.shape[0]
        kept = (cond_dict['x0_hand'].reshape(B, -1).abs().sum(dim=1) > 0).float()
        terms = self._add_physics_losses(terms, pred, x_t, t, x_0, x0_hand, touch, kept)
        if self.use_visual_losses and (self.lambda_presence > 0 or self.lambda_carving > 0):
            terms = self._add_visual_losses(terms, pred, x_t, t, x_0,
                                            mask_obj, mask_hand, kept)

        target_mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        distill_mse_per_instance = np.array([
            F.mse_loss(pred[i], teacher_pred[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {
                    "mse": target_mse_per_instance[time_bin == i].mean(),
                    "distill_mse": distill_mse_per_instance[time_bin == i].mean(),
                }

        return terms, {}
