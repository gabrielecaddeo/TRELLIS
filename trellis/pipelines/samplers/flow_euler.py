from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
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
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

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
            model.eval()
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
            for t, t_prev in tqdm(t_pairs, desc=f"Optimization Step {cnt + 1}", disable=not verbose):

                x = checkpoint.checkpoint(
        lambda x_: euler_step(x_, t, t_prev, model, cond, kwargs),
        x
    )
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
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("NaNs/Infs in z after decoder!")
                return z, torch.tensor(1e5, device=x.device)
            
            # with torch.no_grad():
            # torch.save(torch.argwhere(z<=0)[:, [0, 2, 3, 4]].int(), '/home/user/TRELLIS/vectors_test/0000'+str(cnt)+'_vector.pt')
            # torch.save(z, '/home/user/TRELLIS/vectors_test/0000'+str(cnt)+'_vector.pt')
            active_voxels = ((z < 0) & (target < 0)).sum()
            print(f"Active intersecting voxels: {active_voxels.item()}")
            scale=1
            relu_pred = F.relu(-z*scale)
            relu_target = F.relu(-target*scale)
            loss = (relu_pred*relu_target).sum()
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
            clip_grad_norm_(sample, 10.0)

            if verbose:
                print(f'Iter {cnt}: Loss {loss.item():.4f}')    
            return loss

        
        sample = noise.detach().clone()
        decoder.eval()
        model.eval()
        
        sample.requires_grad = True
        # optimizer = torch.optim.Adam([sample], lr=1e-2)
        optimizer = torch.optim.LBFGS([sample], max_iter=5, lr=1e-1, line_search_fn='strong_wolfe')
        best_sample = sample.detach().clone()
        best_loss = float('inf')
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        with torch.no_grad():
            loss_before = loss_fn(model, sample, decoder, target, t_pairs, verbose, cond, **kwargs)[1].item()
            for scale in [1e-2, 5e-2, 1e-1, 2e-1, 1]:
                sample_perturbed = sample + scale * torch.randn_like(sample)
                loss_after = loss_fn(model, sample_perturbed, decoder, target, t_pairs, verbose=False, cond=cond, **kwargs)[1].item()
       
                print(f"Loss delta after perturbing sample: {loss_after - loss_before}")
        for i in range(50):
            
            loss = optimizer.step(closure)
            if loss < best_loss:
                best_loss = loss
                best_sample = sample.detach().clone()
            
        sample_opt = best_sample.detach()
        with torch.no_grad():
            out_opt, _ = loss_fn(model, sample, decoder, target, t_pairs, verbose, cond, **kwargs)
        return sample_opt, out_opt




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
