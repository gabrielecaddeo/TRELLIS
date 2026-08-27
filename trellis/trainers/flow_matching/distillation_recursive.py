"""P4 recursive-student distillation trainer (EVAL_GUIDANCE.md §7.19).

Extends the audited distillation trainer with a prior-latent conditioning
channel: the dataset supplies `prior_sdf` (warped median fusion of OTHER views
of the same grasp — simulated streaming) and `prior_keep`; this trainer encodes
the SDF through the FROZEN sparse-structure VAE encoder into a 16^3x8 latent and
hands it to the model as cond['x0_prior'] (zero-init input_layer_prior branch,
use_prior=True in the model config). Losses are unchanged — distill + target +
physics (+ visual if enabled).

Prior gating stack (all multiplicative):
  - dataset prior_dropout (~30%): preserves the single-view mode, anti-copy;
  - CFG drop: a sample whose conditioning was CFG-dropped also loses the prior
    (the unconditional branch must stay measurement-free). Recovered post-hoc
    from x0_hand == 0, same trick as the physics gating.
The teacher never sees the prior key it does not know — extra cond keys are
ignored by its forward.
"""
import json
import os

import torch

from ... import models
from .distillation import ImageConditionedFlowMatchingCFGDistillationTrainerConditioned


class ImageConditionedFlowMatchingCFGDistillationRecursiveTrainerConditioned(
    ImageConditionedFlowMatchingCFGDistillationTrainerConditioned
):
    def __init__(
        self,
        *args,
        use_prior_latent: bool = True,
        ss_enc_ckpt: str = "ema0.9999_step0300000",
        **kwargs,
    ):
        self.use_prior_latent = use_prior_latent
        self.ss_enc_ckpt = ss_enc_ckpt
        self.ss_enc = None
        super().__init__(*args, **kwargs)
        # AFTER super().__init__: ss_dec_path is only assigned by
        # flow_matching.__init__ after its own super() call returns, so it does
        # not exist yet inside init_models_and_more (the smoke of job 629
        # failed exactly there) — same reason _loading_ss_dec runs post-init.
        if self.use_prior_latent:
            self._loading_ss_enc()

    def _loading_ss_enc(self):
        """Frozen VAE encoder from the same dir as ss_dec (the latents' VAE)."""
        assert self.ss_dec_path is not None, \
            "P4 trainer needs ss_dec_path (the VAE dir also holding the encoder)"
        cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
        enc = getattr(models, cfg['models']['encoder']['name'])(**cfg['models']['encoder']['args'])
        ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'encoder_{self.ss_enc_ckpt}.pt')
        enc.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        self.ss_enc = enc.cuda().eval()
        for p in self.ss_enc.parameters():
            p.requires_grad_(False)
        if self.is_master:
            print(f"Loaded frozen ss_enc for prior latent: {ckpt_path}")

    def _encode_prior(self, prior_sdf, prior_keep):
        with torch.no_grad():
            z = self.ss_enc(prior_sdf.float())          # posterior mean, [B,8,16,16,16]
        keep = (prior_keep.float().view(-1) if prior_keep is not None
                else torch.ones(z.shape[0], device=z.device))
        return z, keep

    def get_cond(self, cond, mask_hand, mask_obj, cond_mask, x0_hand, touch,
                 prior_sdf=None, prior_keep=None, **kwargs):
        cond_dict = super().get_cond(cond, mask_hand, mask_obj, cond_mask,
                                     x0_hand, touch, **kwargs)
        if self.use_prior_latent and prior_sdf is not None:
            z, keep = self._encode_prior(prior_sdf, prior_keep)
            # CFG-dropped samples (x0_hand zeroed by the CFG mixin) lose the
            # prior too — same post-hoc mask as the physics gating.
            cfg_kept = (cond_dict['x0_hand'].reshape(z.shape[0], -1)
                        .abs().sum(dim=1) > 0).float()
            keep = keep * cfg_kept
            cond_dict['x0_prior'] = z * keep.view(-1, 1, 1, 1, 1)
            cond_dict['prior_keep'] = keep
        return cond_dict

    def get_inference_cond(self, cond, mask_hand, mask_obj, cond_mask, x0_hand,
                           touch, prior_sdf=None, prior_keep=None, **kwargs):
        args = super().get_inference_cond(cond, mask_hand, mask_obj, cond_mask,
                                          x0_hand, touch, **kwargs)
        if self.use_prior_latent and prior_sdf is not None:
            z, keep = self._encode_prior(prior_sdf, prior_keep)
            args['cond']['x0_prior'] = z * keep.view(-1, 1, 1, 1, 1)
            args['cond']['prior_keep'] = keep
            args['neg_cond']['x0_prior'] = torch.zeros_like(z)
            args['neg_cond']['prior_keep'] = torch.zeros_like(keep)
        return args
