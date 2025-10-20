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
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        
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
        alpha_vel: float = 100000,            # physics guidance strength
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
        print('inside 2')
        model.eval()
        decoder.eval()
        for p in model.parameters():   # model weights frozen for sampling
            p.requires_grad_(False)
        for p in decoder.parameters(): # decoder weights frozen, but we still need grads w.r.t. x
            p.requires_grad_(False)

        # time schedule (same as yours, with rescale)
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps)]

        # working latent
        x = noise.detach().clone()

        # (optional) clamp/normalize hand SDF for stability
        # e.g., truncate to a narrow band and scale to [-1, 1]
        # tau = 8.0
        # hand_sdf = hand_sdf.clamp(-tau, tau) / tau
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        alpha_vel = 5000
        print('alpha=', alpha_vel)
        for (t, t_prev) in t_pairs:
            print('inside 3')
            # 1) base vector field v_theta(x,t | cond)
            x = x.detach().requires_grad_(True)  # grads w.r.t. x
            v = self._inference_model(model, x, t, cond, **kwargs)  # shape = x
            # pred_x_prev = x_t - (t - t_prev) * v
            # 2) decode SDF and build physics energy
            #    - interpenetration: penalize inside both hand and obj (negative SDFs)
            S_obj = decoder(x)                                  # [B, 1, 64, 64, 64]
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
            print('E_inter', E_inter.item())
            # print('Loss'E_inter.item())
            # 3) physics gradient w.r.t. latent
            g = torch.autograd.grad(E, x, retain_graph=False, create_graph=False)[0]
            print("Grads",g.abs().mean().item(), g.abs().max().item())
            # 4) guided Euler update: x_{t-Δ} = x_t + Δt * (v - α ∇E)
            dt = t_prev - t
            with torch.no_grad():
                x = x + dt * (v - alpha_vel * g)
            ret.pred_x_t.append(x)
            
                
            if verbose and (not torch.isfinite(x).all()):
                print("Warning: non-finite values in x during guided Euler.")

        # decode final SDF once
        with torch.no_grad():
            final_sdf = decoder(x)

        if save_path is not None:
            torch.save(final_sdf, save_path)
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