
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchdiffeq import odeint, odeint_adjoint

from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from torch.nn.utils import clip_grad_norm_
import torch.utils.checkpoint as checkpoint
def _set_requires_grad(module, flag: bool):
    """Temporarily change requires_grad for all params; returns old flags."""
    old = []
    for p in module.parameters():
        old.append(p.requires_grad)
        p.requires_grad_(flag)
    return old

def _restore_requires_grad(module, old_flags):
    for p, f in zip(module.parameters(), old_flags):
        p.requires_grad_(f)

def _norm_per_sample(g: torch.Tensor, eps=1e-8):
    # g: [B, ...]
    B = g.shape[0]
    gn = g.view(B, -1).norm(dim=1).clamp_min(eps)
    return g / gn.view(B, *([1] * (g.ndim - 1))), gn

class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        # FIX: The original model call `model(x_t, t, cond)` does not expect extra kwargs
        # like `neg_cond` or `cfg_strength`, which were causing TypeErrors.
        # This fix ensures only `cond` is passed, making the method robust to
        # being called from different contexts (like CFG or our optimization loop)
        # without crashing. We explicitly pass only the arguments the model expects.
       

        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        ## Previous implementation
        # if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
        #     cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        B = x_t.shape[0]
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                # Broadcast cond from batch 1 to B if needed
                if cond.shape[0] == 1 and B > 1:
                    cond = cond.repeat(B, *([1] * (cond.ndim - 1)))

            elif isinstance(cond, dict):
                # Broadcast each tensor entry with batch dim 1 -> B
                new_cond = {}
                for k, v in cond.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1 and B > 1:
                        new_cond[k] = v.repeat(B, *([1] * (v.ndim - 1)))
                    else:
                        new_cond[k] = v
                cond = new_cond

        
        return model(x_t, t, cond)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
    


    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret
    
    def sample_velocity(
        self,
        model,
        noise,                         # latent init: [B, 8, 16, 16, 16]
        decoder,                       # frozen decoder: latent -> SDF [B, 1, 64, 64, 64]
        hand_sdf,                      # SDF of the hand on 64^3 (same shape as decoder output)
        cond: dict | None = None,      # your conditioning (masks etc.)
        steps: int = 50,
        rescale_t: float = 1.0,
        alpha_vel: float = 5000,       # physics guidance strength; UNCALIBRATED for the corrected update -- sweep before trusting. 0 disables guidance.
        delta: float = 2.0,            # contact band (voxels); set 0 to disable
        beta: float = 0.0,             # contact guidance weight (0 = off)
        save_path: str | None = None,  # where to torch.save the final SDF (optional)
        verbose: bool = True,
        **kwargs
    ):
        """
        Guided Euler sampling: at each step, nudge the vector field with the
        gradient of a physics energy computed via the *frozen* decoder.

        The energy is evaluated on the clean-object estimate x0_hat obtained from the
        current velocity via _v_to_xstart_eps (the decoder is trained on clean latents,
        so decoding the noisy interpolant x_t gives meaningless SDFs at high t -- the
        same defect as training finding B). Guidance is additionally skipped for the
        first `guidance_skip` steps, where x0_hat is still noise-dominated.

        Set alpha_vel=0 to recover plain (unguided) Euler sampling exactly.

        Returns:
            {"samples": latent, "pred_x_t": [...], "pred_x_0": [...]}
        """
        model.eval()
        decoder.eval()
        for p in model.parameters():   # model weights frozen for sampling
            p.requires_grad_(False)
        for p in decoder.parameters(): # decoder weights frozen, but we still need grads w.r.t. x
            p.requires_grad_(False)

        guidance_skip = int(kwargs.pop("guidance_skip", 5))

        # time schedule (same as sample(), with rescale)
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

        # working latent
        x = noise.detach().clone()

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for step_i, (t, t_prev) in enumerate(t_pairs):
            # 1) base vector field v_theta(x,t | cond). Computed without grad: the
            #    energy gradient below flows only through x_t's direct affine path to
            #    x0_hat (DPS-style approximation), not back through the model.
            with torch.no_grad():
                v = self._inference_model(model, x, t, cond, **kwargs)  # shape = x

            guidance_on = alpha_vel > 0 and step_i >= guidance_skip
            if guidance_on:
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    pred_x_0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)

                    # 2) physics energy on the decoded clean-object estimate
                    S_obj = decoder(pred_x_0)                       # [B, 1, 64, 64, 64]
                    E_inter = (F.relu(-hand_sdf) * F.relu(-S_obj)).sum()

                    # optional: contact encouragement near hand surface (outside hand)
                    if beta > 0.0 and delta > 0.0:
                        band = (hand_sdf.abs() < delta) & (hand_sdf >= 0)
                        if band.any():
                            # bring object surface (S_obj ~ 0) near hand surface in the band
                            E_contact = F.smooth_l1_loss(S_obj[band], torch.zeros_like(S_obj[band]))
                        else:
                            E_contact = S_obj.new_zeros(())
                        E = E_inter + beta * E_contact
                    else:
                        E = E_inter

                    # 3) physics gradient w.r.t. the latent
                    g = torch.autograd.grad(E, x, retain_graph=False, create_graph=False)[0]
            else:
                with torch.no_grad():
                    pred_x_0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)
                g = torch.zeros_like(x)

            # 4) guided Euler update: x_{t_prev} = x_t - (t - t_prev) * (v + alpha * grad E)
            #    (t_prev < t, so dt > 0; the -alpha*g term descends the energy)
            dt = t - t_prev
            with torch.no_grad():
                x = x - dt * (v + alpha_vel * g)

            ret.pred_x_t.append(x.detach())
            ret.pred_x_0.append(pred_x_0.detach())

            if verbose and (not torch.isfinite(x).all()):
                print("Warning: non-finite values in x during guided Euler.")

        # decode final SDF once
        with torch.no_grad():
            final_sdf = decoder(x)

        if save_path is not None:
            torch.save(final_sdf, save_path)
        ret.samples = x.detach()
        return ret
    
    def sample_velocity_conditioned(
        self,
        model,
        noise,                         # latent init: [B, 8, 16, 16, 16]
        decoder,                       # frozen decoder: latent -> SDF [B, 1, 64, 64, 64]
        cond: dict | None = None,      # your conditioning (masks etc.)
        neg_cond: dict | None = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        alpha_vel: float = 10,         # physics guidance strength (matches the operative inference-repo value); 0 disables guidance exactly
        delta: float = 2.0,            # contact band (voxels); set 0 to disable
        beta: float = 0.0,             # contact guidance weight (0 = off)
        save_path: str | None = None,  # where to torch.save the final SDF (optional)
        verbose: bool = True,
        **kwargs
    ):
        """
        Guided Euler sampling: at each step, nudge the vector field with the
        gradient of a physics energy computed via the *frozen* decoder.

        Returns:
            {"samples": latent, "final_sdf": sdf}
        """
        # --- setup ---
        # ---- 0. Setup & extract needed pieces ----
        model.eval()
        decoder.eval()

        # we do NOT want to train model/decoder weights during sampling
        #for p in model.parameters():
        #    p.requires_grad_(False)
        #for p in decoder.parameters():
        #    p.requires_grad_(False)

        # unpack condition bundle
        cond = cond or {}
        pos_cond = cond
        
        print(type(pos_cond))
        x0_hand  = pos_cond.get("x0_hand", None)     # latent for hand
        touch    = pos_cond.get("touch",   None)     # [B, 1, 64, 64, 64] contact mask

        # precompute sdf_hand ONCE (no grad wrt hand)
        if x0_hand is not None:
            with torch.no_grad():
                sdf_hand = decoder(x0_hand)     # [B, 1, 64, 64, 64]
                #torch.save(sdf_hand, '/home/user/TRELLIS/coords_asym_velocity/hand_sdf.pt')
        else:
            sdf_hand = None

        # time schedule (same as your sample())
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

        # working latent
        x = noise.detach().clone()
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        lambda_inter = 500
        lambda_contact = 50
        guidance_skip = int(kwargs.pop("guidance_skip", 5))  # x0_hat is noise-dominated in the first steps
        for step_i, (t, t_prev) in enumerate(t_pairs):
            guidance_on = alpha_vel > 0 and step_i >= guidance_skip
            # 1) base vector field v_theta(x,t | cond)
            with torch.no_grad():
                v = self._inference_model(model, x, t, pos_cond, neg_cond, **kwargs)  # shape = x
            if guidance_on:
                print('guidance')
                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)  # grads w.r.t. x
                    pred_x_0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)
                    
                    # ---- 2) build E_sdf(x) ~ lambda_ni * ni_loss + lambda_contact * contact_loss ----
                    E = x.new_zeros([])
                    sdf_obj = decoder(pred_x_0)              # [B, 1, 64, 64, 64] (requires grad wrt x)
                    # save intermediate SDFs for visualization/debugging
                    if save_path is not None:
                        torch.save(sdf_obj, save_path)
                    obj_inside  = torch.clamp(-sdf_obj,  0.0, 0.1)
                    hand_inside = torch.clamp(-sdf_hand, 0.0, 0.1)                                  # [B, 1, 64, 64, 64]
                    interpenetration = obj_inside * hand_inside
                    pen_mask = (obj_inside > 0) & (hand_inside > 0)
                    
                    B = interpenetration.shape[0]
                    num = (interpenetration * pen_mask).view(B, -1).sum(dim=1)
                    den = pen_mask.view(B, -1).sum(dim=1).clamp_min(1)
                    ni_per_sample = num / den
                    ni_loss = ni_per_sample.mean()
                    
                    E = E + lambda_inter * ni_loss
                    contact_mask = touch[:, 0]                 # [B, 64, 64, 64]
                    contact_sdf  = contact_mask * sdf_obj.abs()
                    
                    B = contact_sdf.shape[0]
                    num   = contact_sdf.view(B, -1).sum(dim=1)
                    denom = contact_mask.view(B, -1).sum(dim=1).clamp_min(1)
                    
                    per_sample_loss = num / denom
                    contact_loss = per_sample_loss.mean()

                    E = E + lambda_contact * contact_loss
                    # print('Loss'E_inter.item())
                    # 3) physics gradient w.r.t. latent
                    
                    g = torch.autograd.grad(E, x, retain_graph=False, create_graph=False)[0]
            else:
                # no physics term active
                pred_x_0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)
                g = torch.zeros_like(x)
            
            # with torch.no_grad():
            #     pen_vox = pen_mask.sum().item()
            #     touch_vox = touch[:,0].sum().item() if touch is not None else -1
            # print(f"pen_vox={pen_vox}, touch_vox={touch_vox}, ni_loss={ni_loss.item():.3e}, contact={contact_loss.item():.3e}")
            # if verbose:
            #     # this is useful for checking stability
            #     g_mean = g.abs().mean().item()
            #     g_max  = g.abs().max().item()
            #     print("alpha_vel", alpha_vel, type(alpha_vel))
            #     print("v_abs_max", v.abs().max().item(), "g_abs_max", g.abs().max().item())

            #     print(f"t={t:.4f}, E={E.item():.4e}, |g|_mean={g_mean:.4e}, |g|_max={g_max:.4e}")

            # ---- 4) guided Euler update: x_{t-Δ} = x_t + Δt * (v - α ∇E) ----
            dt =  t-t_prev   # note: t_prev < t, so dt is negative
            with torch.no_grad():
                x = x - dt * (v + alpha_vel * g)

            ret.pred_x_t.append(x.detach())
            ret.pred_x_0.append(pred_x_0.detach())

            v_step = (dt * v).norm().item()
            g_step = (dt * alpha_vel * g).norm().item()
            x_norm = x.norm().item()

            #print(f"dt={dt:.4f} |x|={x_norm:.3e} |dt*v|={v_step:.3e} |dt*alpha*g|={g_step:.3e} ratio={g_step/(v_step+1e-12):.3e}")
            #print("finite after update?", torch.isfinite(x).all().item(), "max|x|", x.abs().max().item())
            #print("pred_x_0.requires_grad", pred_x_0.requires_grad)
            # print("sdf_obj.requires_grad", sdf_obj.requires_grad)
            # print("E.requires_grad", E.requires_grad)
            #if verbose and (not torch.isfinite(x).all()):
                #print("Warning: non-finite values in x during guided Euler.")
                # you might want to break or clamp here in practice
        
        # decode final SDF once
        with torch.no_grad():
            final_sdf = decoder(x)
        #save_path='/home/user/TRELLIS/coords_asym_velocity/final_sdf.pt'
        if save_path is not None:
            torch.save(final_sdf, save_path)
            
        ret.samples = x.detach()
        return ret


    # ------------------------------------------------------------------
    # Improved physics guidance (greedy) and complete OC-Flow.
    # Both use _physics_energy_per_sample below, so a/b comparisons isolate
    # the optimizer (greedy vs trajectory-level OC) from the cost definition.
    # ------------------------------------------------------------------
    @staticmethod
    def _physics_energy_per_sample(sdf_obj, sdf_hand, touch,
                                   contact_floor=0.011, ni_margin=0.0,
                                   contact_band=0.05):
        """Per-sample (ni, contact) physics energies for sampling-time guidance.

        Fixes over the original guidance energy:
          - NI is normalized by hand-interior mass (extent-sensitive): removing
            penetrated volume reduces it, not just making penetration shallower.
            relu() makes it exactly zero -- gradient included -- for samples with
            no penetration, so NI guidance is inherently violation-triggered.
          - Contact is hinged at the measured decoder floor: a *perfect* latent
            scores ~0.011 mean |sdf| at the annotated contact voxels through this
            frozen decoder (TEACHER_RETRAIN.md §8.1), so pulling below that is
            over-sharpening past the truth. The hinge also zeroes the term (and
            its gradient) on already-correct samples -> violation-triggered.
          - Contact voxels inside the hand are excluded and |sdf| is band-limited,
            as in sample_velocity_conditioned_oc2.
        """
        B = sdf_obj.shape[0]
        hand_in = (sdf_hand < -ni_margin).float()                    # [B,1,D,H,W]
        obj_pen = F.relu(ni_margin - sdf_obj)                        # depth of violation
        ni_ps = (obj_pen * hand_in).view(B, -1).sum(dim=1) \
            / hand_in.view(B, -1).sum(dim=1).clamp_min(1.0)

        outside = (sdf_hand[:, 0] > 0).float()                       # [B,D,H,W]
        contact_w = touch[:, 0].float() * outside
        abs_sdf = sdf_obj[:, 0].abs().clamp(max=contact_band)
        c_raw = (contact_w * abs_sdf).view(B, -1).sum(dim=1) \
            / contact_w.view(B, -1).sum(dim=1).clamp_min(1.0)
        contact_ps = F.relu(c_raw - contact_floor)
        return ni_ps, contact_ps

    def sample_guided_v2(
        self,
        model,
        noise,
        decoder,
        cond: dict | None = None,
        neg_cond: dict | None = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        # guidance strength: correction norm = rho * (1-t)^time_power * ||v||,
        # per sample -- relative to the velocity instead of an absolute alpha.
        rho: float = 0.2,
        time_power: float = 1.0,
        guidance_skip: int = 5,
        lambda_inter: float = 500.0,
        lambda_contact: float = 50.0,
        contact_floor: float = 0.011,
        ni_margin: float = 0.0,
        contact_band: float = 0.05,
        verbose: bool = True,
        **kwargs
    ):
        """Greedy (per-step) physics guidance with the fixed energy and a
        relative, t-weighted trust region. DPS-style: the energy is evaluated on
        x0_hat = f(x_t, v) with v detached, so gradients flow only through the
        direct affine path (no backprop through the model). rho=0 recovers plain
        Euler exactly; so does any sample whose energies sit at/below their floors.
        """
        model.eval()
        decoder.eval()
        cond = cond or {}
        x0_hand = cond.get("x0_hand", None)
        touch = cond.get("touch", None)
        assert x0_hand is not None and touch is not None, \
            "sample_guided_v2 needs x0_hand and touch in the positive cond"
        with torch.no_grad():
            sdf_hand = decoder(x0_hand).float()

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

        x = noise.detach().clone()
        B = x.shape[0]
        expand = (B,) + (1,) * (x.ndim - 1)
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for step_i, (t, t_prev) in enumerate(t_pairs):
            with torch.no_grad():
                v = self._inference_model(model, x, t, cond, neg_cond, **kwargs)

            guidance_on = rho > 0 and step_i >= guidance_skip
            if guidance_on:
                with torch.enable_grad():
                    x_var = x.detach().requires_grad_(True)
                    pred_x0, _ = self._v_to_xstart_eps(x_t=x_var, t=t, v=v)
                    sdf_obj = decoder(pred_x0)
                    ni_ps, c_ps = self._physics_energy_per_sample(
                        sdf_obj, sdf_hand, touch,
                        contact_floor=contact_floor, ni_margin=ni_margin,
                        contact_band=contact_band)
                    E = (lambda_inter * ni_ps + lambda_contact * c_ps).sum()
                    g = torch.autograd.grad(E, x_var)[0]
                with torch.no_grad():
                    pred_x0 = pred_x0.detach()
                    g_norm = g.view(B, -1).norm(dim=1)
                    v_norm = v.view(B, -1).norm(dim=1)
                    w = rho * (1.0 - t) ** time_power
                    scale = torch.where(
                        g_norm > 1e-8,
                        w * v_norm / g_norm.clamp_min(1e-8),
                        torch.zeros_like(g_norm))
                    u = g * scale.view(expand)
            else:
                with torch.no_grad():
                    pred_x0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)
                u = torch.zeros_like(x)

            dt = t - t_prev
            with torch.no_grad():
                x = x - dt * (v + u)
            ret.pred_x_t.append(x.detach())
            ret.pred_x_0.append(pred_x0)

        ret.samples = x.detach()
        return ret

    def sample_oc_flow(
        self,
        model,
        noise,
        decoder,
        cond: dict | None = None,
        neg_cond: dict | None = None,
        steps: int = 25,
        rescale_t: float = 1.0,
        # outer-loop optimal control
        n_outer: int = 4,
        eta_u: float = 0.5,          # control learning rate (per-sample normalized)
        u_decay: float = 1.0,        # 1.0 = no control regularization between iters
        u_max_ratio: float = 0.3,    # trust region: ||u_i|| <= ratio * ||v_i||
        use_model_jacobian: bool = True,
        lambda_inter: float = 500.0,
        lambda_contact: float = 50.0,
        contact_floor: float = 0.011,
        ni_margin: float = 0.0,
        contact_band: float = 0.05,
        verbose: bool = True,
        **kwargs
    ):
        """Complete OC-Flow (discrete adjoint): optimize per-step controls u_i over
        the WHOLE trajectory against the terminal physics cost, with outer
        iterations.

            rollout:   x_{i+1} = x_i - dt_i * (v(x_i, t_i) + u_i)
            terminal:  E = E_phys(decoder(x_K))              (x_K is the clean sample)
            adjoint:   a_K = dE/dx_K,
                       a_i = a_{i+1} - dt_i * (dv/dx|_{x_i})^T a_{i+1}   (VJP)
            update:    u_i <- u_decay * u_i + eta_u * dt_i * a_{i+1}, trust-regioned

        Each outer iteration costs one rollout plus one VJP (model fwd+bwd) per
        step, so it is ~2*n_outer times the price of greedy guidance. Set
        use_model_jacobian=False to skip the VJPs (a_i = a_K for all i), which is
        the FlowGrad-style straight-through approximation.

        Uses the same fixed physics energy as sample_guided_v2, so differences in
        the a/b are attributable to the optimizer, not the cost.
        """
        model.eval()
        decoder.eval()
        cond = cond or {}
        x0_hand = cond.get("x0_hand", None)
        touch = cond.get("touch", None)
        assert x0_hand is not None and touch is not None, \
            "sample_oc_flow needs x0_hand and touch in the positive cond"
        with torch.no_grad():
            sdf_hand = decoder(x0_hand).float()

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]
        dts = [t - tp for (t, tp) in t_pairs]

        x0 = noise.detach().clone()
        B = x0.shape[0]
        expand = (B,) + (1,) * (x0.ndim - 1)
        u = [torch.zeros_like(x0) for _ in range(steps)]

        def energy(x_final_var):
            sdf_obj = decoder(x_final_var)
            ni_ps, c_ps = self._physics_energy_per_sample(
                sdf_obj, sdf_hand, touch,
                contact_floor=contact_floor, ni_margin=ni_margin,
                contact_band=contact_band)
            return (lambda_inter * ni_ps + lambda_contact * c_ps).sum(), ni_ps, c_ps

        x_final = None
        for it in range(n_outer + 1):
            # ---- rollout with current controls (also the final, u-frozen pass) ----
            xs, v_norms = [], []
            x = x0
            with torch.no_grad():
                for i, (t, t_prev) in enumerate(t_pairs):
                    xs.append(x)
                    v = self._inference_model(model, x, t, cond, neg_cond, **kwargs)
                    v_norms.append(v.view(B, -1).norm(dim=1))
                    x = x - dts[i] * (v + u[i])
            x_final = x
            if it == n_outer:
                break

            # ---- terminal cost and its gradient ----
            with torch.enable_grad():
                xK = x_final.detach().requires_grad_(True)
                E, ni_ps, c_ps = energy(xK)
                aK = torch.autograd.grad(E, xK)[0]
            if verbose:
                print(f"  [oc-flow] outer {it + 1}/{n_outer}: "
                      f"E={float(E):.5g} ni={float(ni_ps.mean()):.4g} "
                      f"contact={float(c_ps.mean()):.4g}")
            if float(E) == 0.0:
                break  # every sample at/below its floor -- nothing to optimize

            # ---- adjoint backward + control update ----
            a = aK
            for i in reversed(range(steps)):
                # descent on E: u_i <- u_decay*u_i - eta*dE/du_i, dE/du_i = -dt_i*a_{i+1}
                with torch.no_grad():
                    a_norm = a.view(B, -1).norm(dim=1)
                    step_dir = a * torch.where(
                        a_norm > 1e-8,
                        v_norms[i] / a_norm.clamp_min(1e-8),
                        torch.zeros_like(a_norm)).view(expand)
                    u[i] = u_decay * u[i] + eta_u * dts[i] * step_dir
                    # trust region against this rollout's velocity norm
                    un = u[i].view(B, -1).norm(dim=1)
                    cap = (u_max_ratio * v_norms[i] / un.clamp_min(1e-8)).clamp(max=1.0)
                    u[i] = u[i] * cap.view(expand)

                if i > 0 and use_model_jacobian:
                    with torch.enable_grad():
                        xi = xs[i].detach().requires_grad_(True)
                        vi = self._inference_model(model, xi, t_pairs[i][0], cond,
                                                   neg_cond, **kwargs)
                        vjp = torch.autograd.grad((vi * a.detach()).sum(), xi)[0]
                    a = a - dts[i] * vjp

        ret = edict({"samples": x_final.detach(), "pred_x_t": [], "pred_x_0": []})
        return ret

    def sample_velocity_conditioned_oc2(
            self,
            model,
            noise,
            decoder,
            cond: dict | None = None,        # pos
            neg_cond: dict | None = None,    # neg
            steps: int = 50,
            rescale_t: float = 1.0,
            cfg_strength: float = 3.0,
        
            # OC / guidance
            guidance_start: int = 10,
            theta_lr: float = 0.2,           # how fast theta follows gradients
            theta_decay: float = 0.9,        # EMA decay
            theta_max_ratio: float = 0.6,    # ||theta|| <= ratio * ||v||
            # gradient scaling (helps when grads are tiny)
            target_g: float = 20.0,          # desired per-sample ||g|| after mixing
            max_g_scale: float = 50.0,       # cap for amplification
            
            # losses
            lambda_inter: float = 500.0,
            lambda_contact: float = 50.0,
            
            # smooth gates / band
            tau: float = 0.05,               # gate sharpness for hand inside/outside
            contact_band: float = 0.05,      # clamp |sdf| in contact to avoid huge far-field influence
            
            # safety
            project_contact_descent: bool = True,  # enforce dirder(contact) <= 0
            verbose: bool = True,
            **kwargs
    ):
        """
        OC-style guidance for flow matching sampling:
        x_{t_prev} = x_t - dt * ( v_theta(x_t,t) + theta_t )
        where theta_t is an auxiliary control updated from ∇_x losses computed through
        pred_x0 = f(x_t, v(x_t,t)) and the frozen decoder.
        
        IMPORTANT SIGN:
        Because update uses x <- x - dt*(...), to *decrease* a loss L, you want
        theta to align with +∇_x L (so that -dt*theta is a descent step).
        """
        
        model.eval()
        decoder.eval()
        cond = cond or {}
        neg_cond = neg_cond or None
        
        # pull hand + touch from positive cond
        x0_hand = cond.get("x0_hand", None)
        touch   = cond.get("touch", None)
    
        # precompute hand sdf once (no grad)
        sdf_hand = None
        if x0_hand is not None:
            with torch.no_grad():
                sdf_hand = decoder(x0_hand)  # [B,1,64,64,64]

        # time schedule t: 1 -> 0
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]
        
        x = noise.detach().clone()
        theta = torch.zeros_like(x)
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        # Freeze weights *temporarily* to save memory, but restore afterwards
        old_model_flags = _set_requires_grad(model, False)
        old_dec_flags   = _set_requires_grad(decoder, False)

        try:
            for step_i, (t, t_prev) in enumerate(t_pairs):
                dt = (t - t_prev)  # positive (since t > t_prev)
                guidance_on = (
                    step_i >= guidance_start
                    and (sdf_hand is not None)
                    and (touch is not None)
                )
                print(step_i)
                if not guidance_on:
                    # No grads anywhere
                    with torch.no_grad():
                        v = self._inference_model(model, x, t, cond, neg_cond, cfg_strength, **kwargs)
                        # if neg_cond is not None and cfg_strength is not None and cfg_strength != 0.0:
                        #     v_neg = self._inference_model(model, x, t, neg_cond, **kwargs)
                        #     v = v_pos + cfg_strength * (v_pos - v_neg)
                        # else:
                        #     v = v_pos

                        theta.mul_(theta_decay)
                        x = x - dt * (v + theta)

                    ret.pred_x_t.append(x.detach())
                    # store pred_x0 for debugging/vis if you want
                    with torch.no_grad():
                        pred_x0, _ = self._v_to_xstart_eps(x_t=x, t=t, v=v)
                    ret.pred_x_0.append(pred_x0.detach())
                    continue

                # --- Guidance ON: grad wrt x only ---
                x_var = x.detach().requires_grad_(True)
                
                # CFG velocity on x_var (so ∂v/∂x exists if you want OC sensitivity)
                v = self._inference_model(model, x_var, t, cond, neg_cond, cfg_strength, **kwargs)
                #if neg_cond is not None and cfg_strength is not None and cfg_strength != 0.0:
                #    v_neg = self._inference_model(model, x_var, t, neg_cond, **kwargs)
                #    v = v_pos + cfg_strength * (v_pos - v_neg)
                #else:
                #    v = v_pos

                # map (x_t, v) -> pred_x0
                pred_x0, _ = self._v_to_xstart_eps(x_t=x_var, t=t, v=v)
                sdf_obj = decoder(pred_x0)  # [B,1,64,64,64]
                
                B = sdf_obj.shape[0]
                
                # -------------------------
                # Non-interpenetration loss (non-saturating)
                # -------------------------
                # hand_in ~ 1 inside hand, 0 outside (smooth)
                hand_in = torch.sigmoid(-sdf_hand / tau)     # [B,1,D,H,W]
                # penalize negative sdf in object (penetration depth)
                obj_pen = torch.relu(-sdf_obj)               # [B,1,D,H,W]
                
                num_ni = (obj_pen * hand_in).view(B, -1).sum(dim=1)
                den_ni = hand_in.view(B, -1).sum(dim=1).clamp_min(1.0)
                ni_loss = (num_ni / den_ni).mean()
                
                # -------------------------
                # Contact loss (only outside hand + banded |sdf|)
                # -------------------------
                outside = torch.sigmoid(sdf_hand / tau)      # ~1 outside
                contact_mask = touch[:, 0]                   # [B,D,H,W]
                contact_w = contact_mask * outside[:, 0]     # [B,D,H,W]
                
                abs_sdf = sdf_obj.abs()
                abs_sdf = torch.clamp(abs_sdf, 0.0, contact_band)  # banded
                contact_sdf = contact_w * abs_sdf[:, 0]            # [B,D,H,W]
                
                num_c = contact_sdf.view(B, -1).sum(dim=1)
                den_c = contact_w.view(B, -1).sum(dim=1).clamp_min(1.0)
                contact_loss = (num_c / den_c).mean()
                
                # gradients wrt x_var (separate)
                g_ni = torch.autograd.grad(ni_loss, x_var, retain_graph=True, create_graph=False)[0]
                g_c  = torch.autograd.grad(contact_loss, x_var, retain_graph=True, create_graph=False)[0]
                
                # normalize per-sample then mix with lambdas (this prevents one term dominating by scale)
                g_ni_n, _ = _norm_per_sample(g_ni)
                g_c_n,  _ = _norm_per_sample(g_c)
                g = (lambda_inter * g_ni_n) + (lambda_contact * g_c_n)
                
                # optional: amplify when small, but cap
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                g_n, gnorm = _norm_per_sample(g)
                # g_n has per-sample norm 1; scale it to target_g
                scale = (target_g / gnorm.view(B, *([1]*(g.ndim-1)))).clamp(max=max_g_scale)
                g = g * scale
                
                # -------------------------
                # Update theta (SIGN!)
                # -------------------------
                # Because x <- x - dt*(v + theta),
                # to do descent on E you want theta ~ +∇E (so -dt*theta is -∇E step).
                with torch.no_grad():
                    theta.mul_(theta_decay).add_(theta_lr * g)

                    # trust region: ||theta|| <= ratio * ||v||
                    v_det = v.detach()
                    v_norm = v_det.view(B, -1).norm(dim=1).clamp_min(1e-8)
                    th_norm = theta.view(B, -1).norm(dim=1).clamp_min(1e-8)
                    max_th = theta_max_ratio * v_norm
                    th_scale = (max_th / th_norm).clamp(max=1.0).view(B, *([1]*(theta.ndim-1)))
                    theta.mul_(th_scale)
                    
                    # -------------------------
                    # Projection: enforce contact descent (correct sign)
                    # -------------------------
                    # Your directional derivative print is:
                    #   dirder_contact = <∇contact, step_dir> where step_dir = - (v + theta)
                    # So contact decreases if dirder_contact <= 0  <=>  <∇c, (v+theta)> >= 0.
                    if project_contact_descent:
                        gc = g_c.detach()
                        u = (v_det + theta)  # u is the thing multiplied by dt in x update
                        
                        gc_flat = gc.view(B, -1)
                        u_flat  = u.view(B, -1)
                        inner = (gc_flat * u_flat).sum(dim=1, keepdim=True)  # <gc, u>
                        gc2   = (gc_flat * gc_flat).sum(dim=1, keepdim=True).clamp_min(1e-8)
                        
                        # If inner < 0 -> dirder_contact = -inner > 0 (uphill), fix it by adding beta*gc to u
                        # since <gc, u + beta*gc> = inner + beta*||gc||^2, choose beta = -inner/||gc||^2
                        beta = (-inner / gc2).clamp_min(0.0)  # only when inner < 0
                        theta.add_(beta.view(B, *([1]*(theta.ndim-1))) * gc)
                        
                        # re-apply trust region after projection
                        th_norm = theta.view(B, -1).norm(dim=1).clamp_min(1e-8)
                        th_scale = (max_th / th_norm).clamp(max=1.0).view(B, *([1]*(theta.ndim-1)))
                        theta.mul_(th_scale)
                        
                    # Euler step (use detached v)
                    x = x - dt * (v_det + theta)

                # logging
                if verbose:
                    with torch.no_grad():
                        # recompute directional derivatives using current u = v + theta
                        u = (v_det + theta)
                        gc = g_c.detach()
                        gni = g_ni.detach()
                        dir_c = (gc.view(B, -1) * (-u).view(B, -1)).sum(dim=1).mean().item()
                        dir_ni = (gni.view(B, -1) * (-u).view(B, -1)).sum(dim=1).mean().item()
                        
                        step_v = (dt * v_det).norm().item()
                        step_th = (dt * theta).norm().item()
                        print(f"step={step_i:02d} dt={dt:.4f} ratio(theta/v)={step_th/(step_v+1e-12):.3f} "
                              f"ni={ni_loss.item():.4e} c={contact_loss.item():.4e} "
                              f"dirder_c={dir_c:.3e} dirder_ni={dir_ni:.3e} "
                              f"contact_w_mean={contact_w.mean().item():.3e} sum={contact_w.sum().item():.3f}")
                        
                ret.pred_x_t.append(x.detach())
                ret.pred_x_0.append(pred_x0.detach())

            ret.samples = x.detach()
            return ret

        finally:
            _restore_requires_grad(model, old_model_flags)
            _restore_requires_grad(decoder, old_dec_flags)
    
    def sample_velocity_conditioned_oc(
            self,
            model,
            noise,
            decoder,
            cond: dict | None = None,        # pos
            neg_cond: dict | None = None,    # neg
            steps: int = 50,
            rescale_t: float = 1.0,
            cfg_strength: float = 3.0,
            # OC / physics
            theta_lr: float = 0.3,
            theta_decay: float = 0.9,
            theta_max_ratio: float = 0.6,
            lambda_inter: float = 100.0,
            lambda_contact: float = 50.0,
            guidance_start: int = 10,
            verbose: bool = True,
            **kwargs
    ):
        model.eval()
        decoder.eval()
        
        cond = cond or {}
        neg_cond = neg_cond or None
        
        # hand + touch should come from POSITIVE condition
        x0_hand = cond.get("x0_hand", None)
        touch   = cond.get("touch", None)
        
        # precompute hand sdf once
        if x0_hand is not None:
            with torch.no_grad():
                sdf_hand = decoder(x0_hand)
        else:
            sdf_hand = None

        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]
        
        x = noise.detach().clone()
        theta = torch.zeros_like(x)
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # freeze weights (still allows grad wrt inputs)
        
        for step_i, (t, t_prev) in enumerate(t_pairs):
            dt = (t - t_prev)  # positive
            guidance_on = (step_i >= guidance_start) and (sdf_hand is not None) and (touch is not None)

            if guidance_on:
                #print('guidance')
                # --- enable grad wrt x only ---
                x_var = x.detach().requires_grad_(True)
                print("step: ",step_i)
                #print("dt: ", dt)
                # CFG velocity computed on x_var (so ∂v/∂x exists)
                
                v = self._inference_model(model, x_var, t, cond, neg_cond, cfg_strength, **kwargs)
                #if (neg_cond is not None) and (cfg_strength is not None) and (cfg_strength != 0.0):
                #    v_neg = self._inference_model(model, x_var, t, cond=neg_cond, **kwargs)
                #    v = (1.0 + cfg_strength) * v_pos - cfg_strength * v_neg
                #else:
                #    v = v_pos

                # pred_x0 must use v (NOT detached) for OC-style sensitivity
                pred_x0, _ = self._v_to_xstart_eps(x_t=x_var, t=t, v=v)
                
                sdf_obj = decoder(pred_x0)
                
                # --- NI loss (same as your inference version) ---
                # obj_inside  = torch.clamp(-sdf_obj,  0.0, 0.2)
                # hand_inside = torch.clamp(-sdf_hand, 0.0, 0.2)
                # inter = obj_inside * hand_inside
                # pen_mask = (obj_inside > 0) & (hand_inside > 0)
        
                B = sdf_obj.shape[0]
                # num = (inter * pen_mask).view(B, -1).sum(dim=1)
                # den = pen_mask.view(B, -1).sum(dim=1).clamp_min(1)
                # ni_loss = (num / den).mean()
                #obj_inside  = torch.relu(-sdf_obj)    # [B,1,D,H,W]
                #hand_inside = torch.relu(-sdf_hand)   # constant wrt x, but ok

                #ni_loss = (obj_inside * hand_inside).mean()

                tau = 0.2
                hand_in = torch.sigmoid(-sdf_hand / tau)     # smooth hand interior weight
                obj_pen = torch.relu(-sdf_obj)              # penetration depth
                
                B = sdf_obj.shape[0]
                num = (obj_pen * hand_in).view(B, -1).sum(dim=1)
                den = hand_in.view(B, -1).sum(dim=1).clamp_min(1.0)
                ni_loss = (num / den).mean()
                # --- contact loss ---
                #contact_mask = touch[:, 0]
                #contact_sdf  = contact_mask * sdf_obj.abs()
                #num_c = contact_sdf.view(B, -1).sum(dim=1)
                #den_c = contact_mask.view(B, -1).sum(dim=1).clamp_min(1)
                #contact_loss = (num_c / den_c).mean()
                outside = torch.sigmoid(sdf_hand / tau)      # ~1 outside, ~0 inside
                contact_mask = touch[:, 0]                   # [B,D,H,W]
                contact_w = contact_mask * outside[:,0]      # prevent rewarding interior contact
                
                contact_sdf = contact_w * sdf_obj.abs()
                num_c = contact_sdf.view(B, -1).sum(dim=1)
                den_c = contact_w.view(B, -1).sum(dim=1).clamp_min(1.0)
                contact_loss = (num_c / den_c).mean()
                E = lambda_inter * ni_loss + lambda_contact * contact_loss
                
                # gradient wrt x_var (includes ∂v/∂x via pred_x0(v(x_var)))
                #g = torch.autograd.grad(E, x_var, retain_graph=False, create_graph=False)[0]
                #g_ni = torch.autograd.grad(ni_loss, x_var, retain_graph=True)[0]
                #g_c  = torch.autograd.grad(contact_loss, x_var, retain_graph=True)[0]
                g_ni = torch.autograd.grad(ni_loss, x_var, retain_graph=True)[0]
                g_c  = torch.autograd.grad(contact_loss, x_var, retain_graph=True)[0]
                print("||g_ni||", g_ni.norm().item(), "||g_c||", g_c.norm().item())
                def norm_per_sample(g):
                    gn = g.view(B, -1).norm(dim=1).clamp_min(1e-8)
                    return g / gn.view(B, *([1]*(g.ndim-1)))
                g = lambda_inter * norm_per_sample(g_ni) + lambda_contact * norm_per_sample(g_c)

                #print("||g_ni||", g_ni.norm().item(), "||g_c||", g_c.norm().item())
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                
                # normalize g per-sample (optional but good)
                g_flat = g.view(B, -1)
                #g_norm = g_flat.norm(dim=1).clamp_min(1e-8).view(B, *([1] * (g.ndim - 1)))
                #g = g / g_norm
                #g_norm = g_flat.norm(dim=1).view(B, *([1] * (g.ndim - 1))).clamp_min(1e-8)
                #max_g = 5.0  # try 1, 5, 10
                #scale = (max_g / g_norm).clamp(max=1.0)
                target_g = 50.0  # try 10, 50, 100
                g_norm = g_flat.norm(dim=1).view(B, *([1]*(g.ndim-1))).clamp_min(1e-8)
                max_scale = 50
                g = g * (target_g / g_norm).clamp(max=max_scale)     # <-- this *amplifies* when g is small
                # update theta + state with no graph accumulation
                with torch.no_grad():
                    #theta.mul_(theta_decay).add_(-theta_lr * g)
                    theta.mul_(theta_decay).add_(+theta_lr * g)
                    # trust region on theta vs v (use v.detach() here safely)
                    v_det = v.detach()
                    v_norm  = v_det.view(B, -1).norm(dim=1).clamp_min(1e-8)
                    th_norm = theta.view(B, -1).norm(dim=1).clamp_min(1e-8)
                    max_th  = theta_max_ratio * v_norm
                    scale = (max_th / th_norm).clamp(max=1.0).view(B, *([1] * (theta.ndim - 1)))
                    theta.mul_(scale)
                    step_v = (dt * v_det).norm().item()
                    step_th = (dt * theta).norm().item()
                    print("ratio theta/v =", step_th/(step_v+1e-12))
                    print("ni_loss: ", ni_loss)
                    print("contact_loss: ", contact_loss)
                    step_dir = -(v_det + theta)  # because x <- x + dt*step_dir
                    # flatten
                    dd_c = (g_c * step_dir).view(B, -1).sum(dim=1).mean().item()
                    dd_ni = (g_ni * step_dir).view(B, -1).sum(dim=1).mean().item()
                    print("dirder contact (should be <0 to decrease):", dd_c,
                          "dirder ni:", dd_ni)
                    print("contact_w mean:", contact_w.mean().item(),
                          "contact_w sum:", contact_w.sum().item(),
                          "outside.mean", outside.mean().item(),
                          "hand_in.mean", hand_in.mean().item())
                    # Euler step (use v_det so we don't keep the graph)
                    # step_dir = -(v_det + theta) should satisfy <g_c, step_dir> <= 0
                    gc = g_c.view(B, -1)
                    step = (v_det + theta).view(B, -1)   # note: step_dir = -step
                    
                    dot = (gc * (-step)).sum(dim=1, keepdim=True)  # <g_c, step_dir>
                    # if dot > 0 => uphill for contact
                    gc2 = (gc * gc).sum(dim=1, keepdim=True).clamp_min(1e-8)
                    
                    # subtract from theta the component that makes it uphill
                    # we adjust "step" by changing theta only
                    # want dot_new = dot - alpha*||gc||^2 <= 0  => alpha >= dot/gc2
                    alpha = (dot / gc2).clamp_min(0.0)   # only if uphill
                    theta = theta - alpha.view(B, *([1]*(theta.ndim-1))) * g_c
                    
                    x = x - dt * (v_det + theta)

            else:
                # --- guidance OFF: no grads anywhere ---
                with torch.no_grad():
                    v = self._inference_model(model, x, t, cond, neg_cond, cfg_strength, **kwargs)
                    #if (neg_cond is not None) and (cfg_strength is not None) and (cfg_strength != 0.0):
                    #    v_neg = self._inference_model(model, x, t, cond=neg_cond, **kwargs)
                    #    v = (1.0 + cfg_strength) * v_pos - cfg_strength * v_neg
                    #else:
                    #    v = v_pos

                    theta.mul_(theta_decay)
                    x = x - dt * (v + theta)

            ret.pred_x_t.append(x.detach())

            if verbose and (step_i % 5 == 0):
                th = theta.abs().mean().item()
                vv = v.abs().mean().item()
                print(f"[{step_i:02d}] t={t:.3f} |theta|mean={th:.2e} |v|mean={vv:.2e} ratio={th/(vv+1e-12):.2e}")


        ret.samples = x.detach()
        return ret

    
    def sample_optimization(
        self,
        model,
        noise,
        decoder,
        target,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        cnt=0
        def euler_step(x, t, t_prev, model, cond, kwargs):
            dt = t_prev - t
            #model.eval()
            # with torch.autograd.set_detect_anomaly(True):
            pred_v = self._inference_model(model, x, t, cond, **kwargs)
            return x + dt * pred_v
        
        def loss_fn(
            model,
            x_t,
            decoder,
            target,
            t_pairs,
            verbose: bool = True,
            cond: Optional[Any] = None,
            **kwargs
            ):

            x = x_t.clone()

            print(f"[LayerNorm32] Input mean: {x.mean().item():.4e}, std: {x.std().item():.4e}")
            
            for t, t_prev in tqdm(t_pairs, desc=f"Optimization Step {cnt + 1}", disable=not verbose):
                x = checkpoint.checkpoint(lambda x_: euler_step(x_, t, t_prev, model, cond, kwargs),x)
                    # x = euler_step(x, t, t_prev, model, cond, kwargs)

            print(f"x stats — min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaNs/Infs in x before decoder!")
                return z, torch.tensor(1e5, device=x.device)
            decoder.eval()

            z = decoder(x)
            # with torch.no_grad():
            #     x_perturb = x + torch.randn_like(x) * 1e-3 
            #     z_perturb = decoder(x_perturb)
            #     print("z delta from small x perturb:", (z - z_perturb).abs().mean().item())
            #     x_perturb = x + torch.randn_like(x) * 1e-1
            #     z_perturb = decoder(x_perturb)
            #     print("z delta from small x perturb:", (z - z_perturb).abs().mean().item())
            print(f"z stats — min: {z.min().item()}, max: {z.max().item()}, mean: {z.mean().item()}")
            print(f"target stats — min: {target.min().item()}, max: {target.max().item()}, mean: {target.mean().item()}")
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("NaNs/Infs in z after decoder!")
                return z, torch.tensor(1e5, device=x.device)
            
            with torch.no_grad():
                torch.save(torch.argwhere(z<=0)[:, [0, 2, 3, 4]].int(), '/home/user/TRELLIS/vectors_test/0000'+str(cnt)+'_vector.pt')
                torch.save(z, '/home/user/TRELLIS/vectors_test/0000'+str(cnt)+'_zvector.pt')
                
            active_voxels = ((z < 0) & (target < 0)).sum()
            print(f"Active intersecting voxels: {active_voxels.item()}")
            scale=1
            relu_pred = F.softplus(-z * scale, beta=5)  # Try beta=10 or 20
            relu_target = F.softplus(-target * scale, beta=5)
            # relu_pred = torch.nn.functional.relu(-z)  # Try beta=10 or 20
            # relu_target = torch.nn.functional.relu(-target)
            print(f"relu_pred stats — min: {relu_pred.min().item()}, max: {relu_pred.max().item()}, mean: {relu_pred.mean().item()}")
            print(f"relu_target stats — min: {relu_target.min().item()}, max: {relu_target.max().item()}, mean: {relu_target.mean().item()}")
            if active_voxels == 0:
                loss = (x*0).sum()  # No active voxels, return zero loss
            else:
                loss = (relu_pred * relu_target).mean()
            return  z, loss
        
        def closure():
            nonlocal cnt
            cnt+=1
            optimizer.zero_grad()
            _, loss = loss_fn(model, sample, decoder, target, t_pairs, verbose, cond, **kwargs)

            if torch.isnan(loss).any():
                return torch.tensor(1e5, device=sample.device)
            
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            
            if cnt == 1:
                grad_vals = sample.grad.flatten().detach().cpu()
                print("Grad stats:")
                print(f"  min: {grad_vals.min().item()}")
                print(f"  max: {grad_vals.max().item()}")
                print(f"  mean: {grad_vals.mean().item()}")
                print(f"  std: {grad_vals.std().item()}")
                print(f"  abs max: {grad_vals.abs().max().item()}")
                print(f"  abs mean: {grad_vals.abs().mean().item()}")
                print("Grad stats — min:", sample.grad.min().item(), 
            "max:", sample.grad.max().item(), 
            "mean:", sample.grad.mean().item())
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"{name} has no gradient!")
            if torch.isnan(sample.grad).any():
                print("NaN detected in gradient!")
                return torch.tensor(1e5, device=sample.device)
            # clip_grad_norm_(sample, 10.0)

            if verbose:
                print(f'Iter {cnt}: Loss {loss.item():.4f}')    
            return loss

        sample = noise.detach().clone()
        decoder.eval()
        model.eval()
        #for p in decoder.parameters():
        #    p.requires_grad = False
        #for p in model.parameters():
        #    p.requires_grad = False
        
        
        sample.requires_grad = True
        # MODIFICATION 3: Use hyperparameters passed from the call site.
        lr = kwargs.get('lr', 1e-3)
        optim_steps = kwargs.get('optim_steps', 50)
        optimizer = torch.optim.Adam([sample], lr=lr)

        best_sample = sample.detach().clone()
        best_loss = float('inf')

        # MODIFICATION 4: Use rescale_t when creating the time sequence.
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        # with torch.no_grad():
        #     loss_before = loss_fn(model, sample, decoder, target, t_pairs, verbose, cond, **kwargs)[1].item()
        #     for scale in [1e-2, 5e-2, 1e-1, 2e-1, 1]:
        #         sample_perturbed = sample + scale * torch.randn_like(sample)
        #         loss_after = loss_fn(model, sample_perturbed, decoder, target, t_pairs, verbose=False, cond=cond, **kwargs)[1].item()
       
        #         print(f"Loss delta after perturbing sample: {loss_after - loss_before}")
        # loss = closure()  # Don't call optimizer.step()
        # loss.backward(retain_graph=True)
        # print("Grad stats post backward:")
        # print(" sample.grad min:", sample.grad.min().item())
        # print(" sample.grad max:", sample.grad.max().item())
        # print(" sample.grad mean:", sample.grad.mean().item())
        # print(" sample.grad std:", sample.grad.std().item())
        # print(" sample.grad isnan:", torch.isnan(sample.grad).any())
        # exit(0)
        # Main optimization loop now uses `optim_steps`.
        for i in range(optim_steps):
            loss = optimizer.step(lambda: closure()) # Pass `i` to the closure
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_sample = sample.detach().clone()
            
        sample_opt = best_sample.detach()
        with torch.no_grad():
            out_opt, _ = loss_fn(model, sample, decoder, target, t_pairs, verbose, cond, **kwargs)
        return edict({"samples": sample_opt, "final_sdf": out_opt})




        #     for t, t_prev in tqdm(t_pairs, desc=f"Optimization Step {i + 1}", disable=not verbose):
        #         out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
        #         sample = out.pred_x_prev
        #         ret.pred_x_t.append(out.pred_x_prev)
        #         ret.pred_x_0.append(out.pred_x_0)

        #     # Compute loss (e.g., L2 loss to a target)
        #     loss = torch.mean((sample - noise) ** 2)
        # for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
        #     out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
        #     sample = out.pred_x_prev
        #     ret.pred_x_t.append(out.pred_x_prev)
        #     ret.pred_x_0.append(out.pred_x_0)
        # ret.samples = sample
        # return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
    

    
    def sample_optimization(
        self,
        model,
        noise,
        decoder,
        target,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample_optimization(model, noise, decoder, target, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)

    def sample_velocity(
        self,
        model,
        noise,
        decoder,
        hand_sdf,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        print('inside')
        return super().sample_velocity(model, noise, decoder, hand_sdf, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
    

    def sample_velocity_conditioned(
        self,
        model,
        noise,
        decoder,
        cond,
        neg_cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # NOTE: everything after `cond` is passed by keyword -- the base signature has
        # neg_cond as its 5th parameter, so positional steps/rescale_t would land in the
        # wrong slots (that bug shipped once: steps=25 arrived as neg_cond).
        return super().sample_velocity_conditioned(
            model, noise, decoder, cond,
            neg_cond=neg_cond, steps=steps, rescale_t=rescale_t,
            cfg_strength=cfg_strength, cfg_interval=cfg_interval,
            verbose=verbose, **kwargs)


# ==============================================================================
#                      Lolli's Playground
# ==============================================================================


def cubic_ramp_loss(pred_sdf: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
    """
    A smooth loss function that penalizes interpenetration using a cubic ramp.
    """
    neg_pred = -pred_sdf
    neg_target = -target_sdf
    cubic_pred = torch.pow(F.relu(neg_pred), 3)
    relu_target = F.relu(neg_target)
    loss = (cubic_pred * relu_target).sum()
    return loss

class FlowGeneralSampler(FlowEulerSampler):
    """
    A unified sampler for D-Flow guidance, supporting multiple solvers and gradient backends.
    """
    def __init__(
        self,
        sigma_min: float,
        gradient_method: str = 'continuous',
        solver_method: str = 'dopri5',
    ):
        super().__init__(sigma_min)
        if gradient_method not in ['discrete', 'continuous']:
            raise ValueError(f"Invalid gradient_method: {gradient_method}.")
        self.gradient_method = gradient_method
        self.solver_method = solver_method
        print(f"Initialized FlowGeneralSampler with: gradient_method='{self.gradient_method}', solver_method='{self.solver_method}'")

    def _get_dynamics(self, model, cond, kwargs):
        def dynamics(t, x):
            _ , _, v = self._get_model_prediction(model, x, t, cond, **kwargs)
            return -v
        return dynamics

    def sample_optimization(
        self,
        model,
        noise,
        decoder,
        target,
        cond: Optional[Any] = None,
        steps: int = 40,
        optim_steps: int = 20,
        lr: float = 1e-2,
        lambda_reg: float = 1e-4,
        debug_logging: bool = False, # NEW: Debug flag
        verbose: bool = True,
        **kwargs
    ):
        model.eval(); decoder.eval()
        #for p in model.parameters(): p.requires_grad_(False)
        #for p in decoder.parameters(): p.requires_grad_(False)
        
        initial_noise = noise.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([initial_noise], lr=lr)
        
        t_span = torch.tensor([1.0, 0.0], device=noise.device)
        
        # BUG FIX (ValueError): Wrap dynamics in an nn.Module for odeint_adjoint
        class Odedynamics(nn.Module):
            def __init__(self, sampler, model, cond, kwargs):
                super().__init__()
                self.sampler = sampler
                self.model = model
                self.cond = cond
                self.kwargs = kwargs
            def forward(self, t, x):
                _, _, v = self.sampler._get_model_prediction(self.model, x, t, self.cond, **self.kwargs)
                return -v
        
        dynamics_module = Odedynamics(self, model, cond, kwargs)
        
        best_loss = float('inf')
        best_sample = initial_noise.detach().clone()

        pbar = tqdm(range(optim_steps), desc=f"D-Flow ({self.gradient_method}/{self.solver_method})", disable=not verbose)
        for i in pbar:
            def closure():
                optimizer.zero_grad()
                
                # --- Solve ODE using the selected backend ---
                if self.gradient_method == 'continuous':
                    
                    final_x_trajectory = odeint_adjoint(dynamics_module, initial_noise, t_span, method=self.solver_method, options={'step_size': 1.0 / steps} if self.solver_method in ['euler', 'midpoint'] else None)
                    final_x = final_x_trajectory[-1]
                
                elif self.gradient_method == 'discrete':
                    if self.solver_method in ['dopri5', 'adams']:
                        raise ValueError(f"Discrete gradient method is incompatible with adaptive solver '{self.solver_method}'.")
                    
                    t_seq = np.linspace(t_span[0].item(), t_span[1].item(), steps + 1)
                    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
                    
                    def euler_step(x, t, t_next):
                        dt = t_next - t
                        _, _, v = self._get_model_prediction(model, x, t, cond, **kwargs)
                        return x - dt * v
                    
                    def midpoint_step(x, t, t_next):
                        dt = t_next - t
                        _, _, v1 = self._get_model_prediction(model, x, t, cond, **kwargs)
                        x_mid = x - (dt / 2.0) * v1
                        t_mid = t + dt / 2.0
                        _, _, v_mid = self._get_model_prediction(model, x_mid, t_mid, cond, **kwargs)
                        return x - dt * v_mid
                        
                    step_fn = {'euler': euler_step, 'midpoint': midpoint_step}.get(self.solver_method)
                    if step_fn is None:
                        raise ValueError(f"Unsupported fixed-step solver for discrete mode: '{self.solver_method}'")

                    x = initial_noise
                    for t, t_next in t_pairs:
                        x = checkpoint.checkpoint(lambda y: step_fn(y, t, t_next), x, use_reentrant=False)
                    final_x = x

                final_z = decoder(final_x)
                loss = cubic_ramp_loss(final_z, target)
                regularization = lambda_reg * torch.mean(initial_noise**2)
                total_loss = loss + regularization
                
                total_loss.backward()

                # --- NEW: Verbose logging block ---
                if debug_logging and initial_noise.grad is not None:
                    grad_vals = initial_noise.grad.flatten().detach().cpu()
                    print(f"\n--- Iter {i+1}/{optim_steps} ---")
                    print(f"  Loss: {total_loss.item():.6f} (Collision: {loss.item():.6f}, Regularization: {regularization.item():.6f})")
                    print(f"  Gradient Stats:")
                    print(f"    min: {grad_vals.min().item():.4e}, max: {grad_vals.max().item():.4e}")
                    print(f"    mean: {grad_vals.mean().item():.4e}, std: {grad_vals.std().item():.4e}")
                    print(f"    abs-max: {grad_vals.abs().max().item():.4e}, abs-mean: {grad_vals.abs().mean().item():.4e}")
                    if torch.isnan(grad_vals).any():
                        print("  WARNING: NaN detected in gradient!")
                # --- End of logging block ---

                if initial_noise.grad is not None:
                    clip_grad_norm_([initial_noise], 10.0)
                
                pbar.set_postfix({"Loss": f"{total_loss.item():.4e}"})
                return total_loss

            loss = optimizer.step(closure)
            
            if loss is not None and loss.item() < best_loss:
                best_loss = loss.item()
                best_sample = initial_noise.detach().clone()
        
        with torch.no_grad():
            # Use the same nn.Module dynamics for the final pass
            final_x_trajectory = odeint(dynamics_module, best_sample, t_span, method=self.solver_method)
            # BUG FIX (RuntimeError): Select the final state at t=0
            final_x_result = final_x_trajectory[-1]
            final_optimized_z = decoder(final_x_result)

        return edict({
            "samples": final_x_result,
            "final_sdf": final_optimized_z,
            "optimized_noise": best_sample,
        })
        
        
class ControlNet(nn.Module):
    """A small MLP to model the time-varying control input `theta(t)`."""
    def __init__(self, data_shape, hidden_dim=32):
        super().__init__()
        flat_dim = np.prod(data_shape)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, flat_dim)
        )
        self.output_shape = data_shape

    def forward(self, t):
        # Ensure t is a scalar tensor for the MLP
        t_in = t.reshape(1, 1)
        control_flat = self.time_mlp(t_in)
        return control_flat.view(self.output_shape)

class OCFlowAdaptiveSampler(FlowEulerSampler):
    """
    Implements guidance via the OC-Flow/FlowGrad paradigm by optimizing a control network.
    """
    def __init__(self, sigma_min: float, solver_method: str = 'dopri5', **kwargs):
        super().__init__(sigma_min)
        self.solver_method = solver_method
        print(f"Initialized OCFlowAdaptiveSampler with: solver_method='{self.solver_method}'")

    def sample_optimization(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        decoder: nn.Module,
        target: torch.Tensor,
        cond: Optional[Any] = None,
        optim_steps: int = 20,
        lr: float = 1e-4,
        gamma: float = 1.0,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        debug_logging: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        model.eval(); decoder.eval()

        control_net = ControlNet(data_shape=noise.shape).to(noise.device)
        optimizer = torch.optim.AdamW(control_net.parameters(), lr=lr)

        t_span = torch.tensor([1.0, 0.0], device=noise.device)

        # We define the augmented dynamics for the ODE solver inside this scope
        # to cleanly capture all necessary variables (model, control_net, cond, etc.).
        class AugmentedDynamics(nn.Module):
            def __init__(self, base_sampler, base_model, control_model, cond_kwargs):
                super().__init__()
                self.base_sampler = base_sampler
                self.base_model = base_model
                self.control_model = control_model
                self.cond_kwargs = cond_kwargs

            def forward(self, t, state):
                x, _ = state # Unpack state: (current_x, current_integral_value)
                # Get velocity `v` from the base model
                _ , _, v = self.base_sampler._get_model_prediction(self.base_model, x, t, **self.cond_kwargs)
                # Get control input `theta` from the control network
                theta = self.control_model(t)
                # ODE for x: dx/dt = - (v + theta)
                dx_dt = -(v + theta)
                # ODE for the running cost: d(integral)/dt = ||theta||^2
                d_integral_dt = torch.sum(theta**2)
                return (dx_dt, d_integral_dt)

        augmented_dynamics = AugmentedDynamics(self, model, control_net, {'cond': cond, **kwargs})
        
        best_loss = float('inf')
        best_control_state_dict = control_net.state_dict()

        pbar = tqdm(range(optim_steps), desc="OC-Flow (Adaptive)", disable=not verbose)
        for i in pbar:
            def closure():
                optimizer.zero_grad()
                # Initial state: (initial_noise, integral_starts_at_zero)
                initial_state = (noise, torch.zeros((), device=noise.device))
                
                # Solve the augmented ODE system using the continuous adjoint method
                final_x, running_cost_integral = odeint_adjoint(
                    augmented_dynamics, initial_state, t_span, method=self.solver_method, rtol=rtol, atol=atol
                )
                final_x = final_x[1] # odeint_adjoint returns a tuple of final states
                running_cost_integral = running_cost_integral[1]

                final_z = decoder(final_x)
                terminal_loss = cubic_ramp_loss(final_z, target)
                total_loss = terminal_loss + (gamma / 2) * running_cost_integral

                total_loss.backward()
                
                if debug_logging:
                    all_grads = []
                    for p in control_net.parameters():
                        if p.grad is not None: all_grads.append(p.grad.flatten())
                    if all_grads:
                        grad_vals = torch.cat(all_grads).detach()
                        print(f"\n--- Iter {i+1}/{optim_steps} ---")
                        print(f"  Loss: {total_loss.item():.6f} (Terminal: {terminal_loss.item():.6f}, Cost: {(gamma / 2) * running_cost_integral.item():.6f})")
                        print(f"  Grad Stats (vs. ControlNet): min={grad_vals.min():.2e}, max={grad_vals.max():.2e}, mean={grad_vals.mean():.2e}, std={grad_vals.std():.2e}")
                
                clip_grad_norm_(control_net.parameters(), 1.0)
                pbar.set_postfix({"Total Loss": f"{total_loss.item():.4e}"})
                return total_loss

            loss = optimizer.step(closure)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_control_state_dict = control_net.state_dict()

        # --- Final non-backprop solve with the best control net found ---
        control_net.load_state_dict(best_control_state_dict)
        with torch.no_grad():
            final_dynamics = AugmentedDynamics(self, model, control_net, {'cond': cond, **kwargs})
            initial_state = (noise, torch.zeros((), device=noise.device))
            # Use the standard `odeint` for the final pass
            final_state_tuple = odeint(final_dynamics, initial_state, t_span, method=self.solver_method)
            final_x_result = final_state_tuple[0][1] # Get x at t=0
            final_optimized_z = decoder(final_x_result)
        
        # BUG FIX: Return a structured edict that the pipeline expects
        return edict({
            "samples": final_x_result,
            "final_sdf": final_optimized_z,
            "optimized_control_net": control_net.state_dict(),
        })
