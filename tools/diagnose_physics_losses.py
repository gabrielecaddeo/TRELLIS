"""
Read-only diagnostics for the teacher's physics losses (non-interpenetration + contact).

Answers four questions about a trained `SparseStructureFlowModelConditioned` checkpoint,
without training anything and without writing to any existing run directory:

  Probe A  Are the NI / contact numbers real, or an artifact of a bad x0 estimate?
           Recomputes both terms on three x0 estimates and compares, bucketed by t.
  Probe B  Gradient attribution: |grad(mse)| vs |grad(ni)| vs |grad(contact)| at fixed t.
  Probe C  Does the model use hand *position*? Swaps / zeroes the hand conditioning.
  Probe D  How much does CFG dropout desync the physics losses?

The math is deliberately a verbatim copy of
`FlowMatchingTrainerConditioned.training_losses` so the numbers are comparable to
`log.txt`; see `physics_terms()` below.

Usage (single GPU, no DDP):

    python tools/diagnose_physics_losses.py \
        --teacher_dir outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP \
        --ckpt        denoiser_ema0.9999_step0054000.pt \
        --data_dir    /projects/gcaddeo/train_flow/TRELLIS/datasets/Leap_Hand \
        --num_batches 200
"""

import os
import re
import sys
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trellis import models, datasets, trainers  # noqa: E402
from trellis.trainers.basic import _total_grad_norm  # noqa: E402

# Matches train.py:16 -- required by torch.use_deterministic_algorithms.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

T_BIN_EDGES = np.linspace(0, 1, 11)
T_BIN_CENTERS = [round(float(c), 2) for c in (T_BIN_EDGES[:-1] + 0.05)]


# --------------------------------------------------------------------------------------
# Trainer subclass: keeps every mixin's __init__ (so encode_image / get_cond / diffuse /
# get_v / _v_to_xstart_eps / ss_dec all come from the real training code) but skips the
# optimizer, EMA, master params and checkpoint load, none of which a read-only probe needs.
# --------------------------------------------------------------------------------------
class DiagnosticTrainer(trainers.ImageConditionedFlowMatchingCFGTrainerConditioned):
    def init_models_and_more(self, **kwargs):
        self.training_models = self.models
        self.model_params = [
            p for m in self.models.values() for p in m.parameters() if p.requires_grad
        ]
        self.master_params = self.model_params
        self.optimizer = None
        self.grad_clip = None

    def prepare_dataloader(self, **kwargs):
        # Deliberately not the min(8, ...) cap from base.py:193-194 -- the sbatch asks
        # for 16 CPUs and each sample decodes a 1024x1024 RGBA render, two masks and two
        # 64^3 float grids, so this is CPU-bound before it is GPU-bound.
        num_workers = kwargs.get("diag_num_workers") or min(
            16, max(2, len(os.sched_getaffinity(0)))
        )
        self.data_sampler = None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.dataset.collate_fn
            if hasattr(self.dataset, "collate_fn")
            else None,
        )

    def __str__(self):
        return f"DiagnosticTrainer(batch_size_per_gpu={self.batch_size_per_gpu})"


# --------------------------------------------------------------------------------------
# The physics terms, copied verbatim from flow_matching.py:660-713 but returning
# per-sample values so they can be bucketed by t.
# --------------------------------------------------------------------------------------
def physics_terms(ss_dec, x0_est, sdf_hand, touch, max_pen=0.1, margin=0.0):
    """Return per-sample NI and contact terms for one x0 estimate.

    `sdf_hand` is passed in (rather than decoded here) because it is identical across
    the three x0 variants and decoding it once per batch saves 2/3 of the decoder cost.

    `margin=0.0` reproduces the un-margined formulation the old teacher trained with;
    `margin=1/64` matches the fix-5 formulation in flow_matching.py:716-727 (NI fires
    within one voxel of the hand surface, and the contact shell is excluded from the
    hand mask so the two terms cannot fight).
    """
    sdf_obj = ss_dec(x0_est).float()  # [B, 1, 64, 64, 64]
    B = sdf_obj.shape[0]

    obj_inside = max_pen * torch.tanh(F.relu(margin - sdf_obj) / max_pen)
    hand_mask = (sdf_hand < -margin).float()

    num = (obj_inside * hand_mask).view(B, -1).sum(dim=1)
    den = hand_mask.view(B, -1).sum(dim=1).clamp_min(1.0)
    ni_per_sample = num / den

    contact_mask = touch[:, 0].float()
    sdf_abs = sdf_obj.abs().squeeze(1)
    contact_sdf = contact_mask * sdf_abs
    c_num = contact_sdf.view(B, -1).sum(dim=1)
    c_den = contact_mask.view(B, -1).sum(dim=1).clamp_min(1.0)
    contact_per_sample = c_num / c_den

    # NOTE: sdf_obj is deliberately not returned -- in probe B it would keep the
    # autograd graph alive across the 10 t-bins and blow up memory.
    return edict(
        ni_per_sample=ni_per_sample,
        contact_per_sample=contact_per_sample,
        # Fraction of the 64^3 grid the decoded shape claims as interior. If this is
        # ~0 the "predicted object" has no volume and the NI term measures nothing.
        frac_obj_inside=(sdf_obj < 0).float().view(B, -1).mean(dim=1),
        frac_hand_inside=(sdf_hand < 0).float().view(B, -1).mean(dim=1),
    )


def time_weights(t, p=2.0):
    """w = (1-t)^p, and the batch normaliser used at flow_matching.py:656-658."""
    w = (1.0 - t).clamp(0.0, 1.0).pow(p)
    return w, w.sum().clamp_min(1e-8)


# --------------------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------------------
class BinnedStats:
    """Accumulates per-sample values into t-bins plus an overall bucket."""

    def __init__(self):
        self._sum = defaultdict(float)
        self._n = defaultdict(int)

    def add(self, key, values, t_bins):
        values = np.asarray(values, dtype=np.float64)
        for v, b in zip(values, t_bins):
            self._sum[(key, int(b))] += float(v)
            self._n[(key, int(b))] += 1
            self._sum[(key, "all")] += float(v)
            self._n[(key, "all")] += 1

    def mean(self, key, b):
        n = self._n[(key, b)]
        return self._sum[(key, b)] / n if n else float("nan")

    def count(self, key, b):
        return self._n[(key, b)]

    def keys(self):
        return sorted({k for k, _ in self._sum})

    def to_dict(self):
        out = {}
        for key in self.keys():
            out[key] = {
                "all": self.mean(key, "all"),
                "n": self.count(key, "all"),
                "by_t_bin": {
                    str(b): {"mean": self.mean(key, b), "n": self.count(key, b)}
                    for b in range(10)
                    if self.count(key, b)
                },
            }
        return out


def print_binned(stats, keys, title):
    print(f"\n--- {title} " + "-" * max(0, 74 - len(title)))
    header = f"{'metric':<28}{'overall':>12}" + "".join(
        f"{c:>9}" for c in T_BIN_CENTERS
    )
    print(header)
    print(f"{'':<28}{'':>12}" + "".join(f"{'t~' + str(c):>9}" for c in T_BIN_CENTERS))
    for key in keys:
        if key not in stats.keys():
            continue
        row = f"{key:<28}{stats.mean(key, 'all'):>12.5g}"
        for b in range(10):
            row += (
                f"{stats.mean(key, b):>9.4g}" if stats.count(key, b) else f"{'-':>9}"
            )
        print(row)


def to_cuda(batch):
    return {
        k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def sample_t_seeded(trainer, batch_size, generator, device):
    """Same schedule as FlowMatchingTrainerConditioned.sample_t, but seeded so the
    paired comparisons in probes A/C see identical t and noise."""
    name = trainer.t_schedule["name"]
    if name == "uniform":
        t = torch.rand(batch_size, generator=generator, device=device)
    elif name == "logitNormal":
        mean = trainer.t_schedule["args"]["mean"]
        std = trainer.t_schedule["args"]["std"]
        t = torch.sigmoid(
            torch.randn(batch_size, generator=generator, device=device) * std + mean
        )
    else:
        raise ValueError(f"Unknown t_schedule: {name}")
    return t.float()


def build_cond(trainer, batch, x0_hand=None, touch=None,
               cond_mask=None, mask_obj=None, mask_hand=None):
    """Call the real `get_cond`, optionally overriding any conditioning pathway.

    Returns (cond_dict, dropped_mask). `dropped_mask` is derived from the returned dict
    rather than from the RNG, so it stays correct regardless of how the CFG mixin draws
    its mask.

    NOTE: `dropped` is read off `x0_hand`, NOT off `cond`. Fix 7 made `get_cond` return an
    all-zeros placeholder for `cond` (no transformer block reads it, so encoding it was a
    wasted ViT-L/14 forward per step). A `cond`-based test would therefore mark *every*
    sample as dropped and silently corrupt probe D's split. The CFG mixin applies one mask
    to all keys, so any per-sample-toggled key works; `x0_hand` is the one the physics
    losses care about.
    """
    cond_dict = trainer.get_cond(
        batch["cond"],
        batch["mask_hand"] if mask_hand is None else mask_hand,
        batch["mask_obj"] if mask_obj is None else mask_obj,
        batch["cond_mask"] if cond_mask is None else cond_mask,
        batch["x0_hand"] if x0_hand is None else x0_hand,
        batch["touch"] if touch is None else touch,
    )
    B = cond_dict["x0_hand"].shape[0]
    dropped = (
        cond_dict["x0_hand"].reshape(B, -1).abs().sum(dim=1) == 0
    ).cpu().numpy()
    return cond_dict, dropped


# --------------------------------------------------------------------------------------
# Probe A -- is the NI / contact number real, or an artifact of a bad x0_pred?
# --------------------------------------------------------------------------------------
@torch.no_grad()
def probe_a(trainer, args):
    print("\n" + "=" * 88)
    print("PROBE A -- NI / contact under three x0 estimates, bucketed by t")
    print("=" * 88)
    trainer.p_uncond = 0.0
    stats = BinnedStats()
    batch_scalars = defaultdict(list)

    it = iter(trainer.dataloader)
    for i in range(args.num_batches):
        try:
            batch = to_cuda(next(it))
        except StopIteration:
            it = iter(trainer.dataloader)
            batch = to_cuda(next(it))

        x_0, x0_hand, touch = batch["x_0"], batch["x0_hand"], batch["touch"]
        B = x_0.shape[0]
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 100003 + i)
        t = sample_t_seeded(trainer, B, g, x_0.device)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        x_t = trainer.diffuse(x_0, t, noise=noise)

        cond_dict, _ = build_cond(trainer, batch)
        pred = trainer.models["denoiser"](x_t, t * 1000, cond_dict)
        target = trainer.get_v(x_0, noise, t)

        mse_per = ((pred - target) ** 2).view(B, -1).mean(dim=1)
        t_np = t.detach().cpu().numpy()
        t_bins = np.clip(np.digitize(t_np, T_BIN_EDGES) - 1, 0, 9)

        sdf_hand = trainer.ss_dec(x0_hand).float()
        variants = {
            # flow_matching.py:662 -- the estimator actually used in training
            "as_trained": (1.0 - trainer.sigma_min) * noise - pred,
            # flow_matching.py:397-410 -- the correct v -> x0 conversion
            "correct": trainer._v_to_xstart_eps(x_t=x_t, t=t, v=pred)[0],
            # the achievable floor
            "gt_floor": x_0,
        }

        stats.add("mse", mse_per.cpu().numpy(), t_bins)
        w, w_sum = time_weights(t)
        results_by_name = {}
        for name, x0_est in variants.items():
            r = physics_terms(trainer.ss_dec, x0_est, sdf_hand, touch,
                              margin=args.ni_margin)
            results_by_name[name] = r
            stats.add(f"ni/{name}", r.ni_per_sample.cpu().numpy(), t_bins)
            stats.add(f"contact/{name}", r.contact_per_sample.cpu().numpy(), t_bins)
            stats.add(f"frac_obj_inside/{name}", r.frac_obj_inside.cpu().numpy(), t_bins)
            # RMS distance of this x0 estimate from the true x_0, in latent units
            err = ((x0_est - x_0) ** 2).view(B, -1).mean(dim=1).sqrt()
            stats.add(f"x0_rmse/{name}", err.cpu().numpy(), t_bins)
            # Batch-level, w-normalised scalars -- these are what log.txt records
            batch_scalars[f"ni_loss_raw/{name}"].append(
                float((w * r.ni_per_sample).sum() / w_sum)
            )
            batch_scalars[f"contact_raw/{name}"].append(
                float((w * r.contact_per_sample).sum() / w_sum)
            )
        stats.add("frac_hand_inside", ((sdf_hand < 0).float().view(B, -1).mean(dim=1)).cpu().numpy(), t_bins)
        batch_scalars["mse"].append(float(mse_per.mean()))
        batch_scalars["x_0_std"].append(float(x_0.std()))

        # Floor-relative terms -- exactly what stage 2 optimises with
        # {ni,contact}_relative: true (flow_matching.py:730-763): the per-sample floor
        # is the ground-truth latent's own score through the same decoder, and the
        # batch reduction is the fix-4 (w * term).mean(), not / w.sum().
        r_gt = results_by_name["gt_floor"]
        for name in ("as_trained", "correct"):
            r = results_by_name[name]
            ni_rel = F.relu(r.ni_per_sample - r_gt.ni_per_sample)
            contact_rel = F.relu(r.contact_per_sample - r_gt.contact_per_sample)
            stats.add(f"ni_rel/{name}", ni_rel.cpu().numpy(), t_bins)
            stats.add(f"contact_rel/{name}", contact_rel.cpu().numpy(), t_bins)
            batch_scalars[f"ni_rel_stage2/{name}"].append(float((w * ni_rel).mean()))
            batch_scalars[f"contact_rel_stage2/{name}"].append(float((w * contact_rel).mean()))

        if (i + 1) % max(1, args.num_batches // 10) == 0:
            print(f"  ... {i + 1}/{args.num_batches} batches")

    print_binned(
        stats,
        ["mse"]
        + [f"{m}/{v}" for m in ("ni", "contact", "frac_obj_inside", "x0_rmse")
           for v in ("as_trained", "correct", "gt_floor")]
        + [f"{m}/{v}" for m in ("ni_rel", "contact_rel")
           for v in ("as_trained", "correct")]
        + ["frac_hand_inside"],
        "Probe A: per-sample means",
    )

    print("\n  Batch-level scalars (directly comparable to log.txt):")
    for k in sorted(batch_scalars):
        v = np.asarray(batch_scalars[k])
        print(f"    {k:<34} {v.mean():>12.6g}  (std {v.std():.3g})")

    print(
        "\n  READ: if frac_obj_inside/as_trained is ~0 while gt_floor is not, the decoded\n"
        "  'predicted object' has no interior and the NI term was measuring nothing.\n"
        "  If contact/gt_floor is already ~= contact/as_trained, the contact loss had no\n"
        "  headroom. x0_rmse/correct should fall sharply as t -> 0; x0_rmse/as_trained\n"
        "  should be flat -- that flatness is finding 2."
    )
    return {"binned": stats.to_dict(),
            "batch_scalars": {k: float(np.mean(v)) for k, v in batch_scalars.items()}}


# --------------------------------------------------------------------------------------
# Probe B -- gradient attribution at fixed t
# --------------------------------------------------------------------------------------
def probe_b(trainer, args):
    print("\n" + "=" * 88)
    print("PROBE B -- gradient norms of mse vs ni vs contact, at fixed t per bin")
    print("=" * 88)
    trainer.p_uncond = 0.0
    denoiser = trainer.models["denoiser"]
    lambda_ni = float(trainer.get_lambda_ni())
    lambda_contact = float(trainer.lambda_contact)
    # A stage-1 config has both lambdas at 0, which would make every physics gradient
    # trivially zero. Measure at unit lambda instead; the ratios scale linearly.
    if lambda_ni == 0.0:
        lambda_ni = 1.0
        print("  lambda_ni is 0 in this config -- measuring at lambda_ni=1.0")
    if lambda_contact == 0.0:
        lambda_contact = 1.0
        print("  lambda_contact is 0 in this config -- measuring at lambda_contact=1.0")
    print(f"  lambda_ni={lambda_ni}  lambda_contact={lambda_contact}  ni_margin={args.ni_margin}")

    results = defaultdict(lambda: defaultdict(list))
    it = iter(trainer.dataloader)

    def grad_norm_of(loss):
        for p in trainer.model_params:
            p.grad = None
        loss.backward(retain_graph=True)
        return float(_total_grad_norm(trainer.model_params))

    for i in range(args.num_batches_grad):
        try:
            batch = to_cuda(next(it))
        except StopIteration:
            it = iter(trainer.dataloader)
            batch = to_cuda(next(it))

        x_0, x0_hand, touch = batch["x_0"], batch["x0_hand"], batch["touch"]
        B = x_0.shape[0]
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 7919 + i)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        with torch.no_grad():
            sdf_hand = trainer.ss_dec(x0_hand).float()
            # Ground-truth floor for the stage-2 relative terms; t-independent, so
            # decoded once per batch.
            r_gt = physics_terms(trainer.ss_dec, x_0, sdf_hand, touch,
                                 margin=args.ni_margin)

        for b, t_val in enumerate(T_BIN_CENTERS):
            t = torch.full((B,), float(t_val), device=x_0.device)
            x_t = trainer.diffuse(x_0, t, noise=noise)
            cond_dict, _ = build_cond(trainer, batch)
            pred = denoiser(x_t, t * 1000, cond_dict)
            target = trainer.get_v(x_0, noise, t)
            w, w_sum = time_weights(t)

            mse = F.mse_loss(pred, target)
            results["gn_mse"][b].append(grad_norm_of(mse))

            for name, x0_est in (
                ("as_trained", (1.0 - trainer.sigma_min) * noise - pred),
                ("correct", trainer._v_to_xstart_eps(x_t=x_t, t=t, v=pred)[0]),
            ):
                r = physics_terms(trainer.ss_dec, x0_est, sdf_hand, touch,
                                  margin=args.ni_margin)
                ni = lambda_ni * ((w * r.ni_per_sample).sum() / w_sum)
                con = lambda_contact * ((w * r.contact_per_sample).sum() / w_sum)
                results[f"gn_ni/{name}"][b].append(grad_norm_of(ni))
                results[f"gn_contact/{name}"][b].append(grad_norm_of(con))
                if name == "correct":
                    # The actual stage-2 objective: floor-relative, margined, fix-4
                    # mean() reduction, at unit lambda ("@1").
                    ni_rel = (w * F.relu(r.ni_per_sample - r_gt.ni_per_sample)).mean()
                    con_rel = (w * F.relu(r.contact_per_sample - r_gt.contact_per_sample)).mean()
                    results["gn_ni_rel@1"][b].append(grad_norm_of(ni_rel))
                    results["gn_contact_rel@1"][b].append(grad_norm_of(con_rel))
                    del ni_rel, con_rel
                del r, ni, con

            for p in trainer.model_params:
                p.grad = None
            del pred, cond_dict, mse
            torch.cuda.empty_cache()

        print(f"  ... {i + 1}/{args.num_batches_grad} batches")

    print(f"\n{'metric':<28}{'overall':>12}" + "".join(f"{c:>10}" for c in T_BIN_CENTERS))
    out = {}
    keys = ["gn_mse"] + [
        f"{m}/{v}" for m in ("gn_ni", "gn_contact") for v in ("as_trained", "correct")
    ] + ["gn_ni_rel@1", "gn_contact_rel@1"]
    for key in keys:
        per_bin = [float(np.mean(results[key][b])) if results[key][b] else float("nan")
                   for b in range(10)]
        overall = float(np.nanmean(per_bin))
        out[key] = {"all": overall, "by_t_bin": dict(zip(map(str, range(10)), per_bin))}
        print(f"{key:<28}{overall:>12.5g}" + "".join(f"{v:>10.3g}" for v in per_bin))

    print(f"\n{'ratio vs mse':<28}{'overall':>12}" + "".join(f"{c:>10}" for c in T_BIN_CENTERS))
    for key in keys[1:]:
        per_bin = [
            out[key]["by_t_bin"][str(b)] / out["gn_mse"]["by_t_bin"][str(b)]
            for b in range(10)
        ]
        out[key + " /mse"] = {"all": float(np.nanmean(per_bin))}
        print(
            f"{key + ' /mse':<28}{float(np.nanmean(per_bin)):>12.5g}"
            + "".join(f"{v:>10.3g}" for v in per_bin)
        )

    print(
        "\n  READ: this is the definitive ratio. If gn_ni/mse and gn_contact/mse are both\n"
        "  <~1e-2 across every t bin even for the 'correct' variant, the physics terms are\n"
        "  not worth reweighting at 64^3 and need redesigning instead."
    )
    return out


# --------------------------------------------------------------------------------------
# Probe C -- does the model use hand position?
# --------------------------------------------------------------------------------------
@torch.no_grad()
def probe_c(trainer, args):
    print("\n" + "=" * 88)
    print("PROBE C -- hand-conditioning ablation (paired, identical t and noise)")
    print("=" * 88)
    trainer.p_uncond = 0.0
    stats = BinnedStats()
    it = iter(trainer.dataloader)

    for i in range(args.num_batches):
        try:
            batch = to_cuda(next(it))
        except StopIteration:
            it = iter(trainer.dataloader)
            batch = to_cuda(next(it))

        x_0, x0_hand, touch = batch["x_0"], batch["x0_hand"], batch["touch"]
        B = x_0.shape[0]
        if B < 2:
            continue
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 104729 + i)
        t = sample_t_seeded(trainer, B, g, x_0.device)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        x_t = trainer.diffuse(x_0, t, noise=noise)
        target = trainer.get_v(x_0, noise, t)
        t_np = t.detach().cpu().numpy()
        t_bins = np.clip(np.digitize(t_np, T_BIN_EDGES) - 1, 0, 9)

        z_ch, z_tc = torch.zeros_like(x0_hand), torch.zeros_like(touch)
        z_cm = torch.zeros_like(batch["cond_mask"])
        z_mo = torch.zeros_like(batch["mask_obj"])
        z_mh = torch.zeros_like(batch["mask_hand"])
        # kwargs to build_cond, per variant.
        variants = {
            "baseline": {},
            # each sample now sees a *different* sample's hand
            "shuffled_hand": dict(x0_hand=torch.roll(x0_hand, 1, dims=0),
                                  touch=torch.roll(touch, 1, dims=0)),
            # the CFG-unconditional hand input
            "zeroed_hand": dict(x0_hand=z_ch, touch=z_tc),
            # --- reference scales: without these the hand numbers are uninterpretable ---
            # "hand costs 0.6% of mse" means nothing until you know what the IMAGE costs.
            # cond_mask is the object-masked image (the pathway that actually reaches the
            # blocks via cross_attn); mask_obj is its attention weighting.
            "zeroed_image": dict(cond_mask=z_cm, mask_obj=z_mo),
            # the 2-D hand-mask branch alone (cross_attn_mask_hand)
            "zeroed_maskhand": dict(mask_hand=z_mh),
            # everything off = the full CFG-unconditional branch; the ceiling for "how much
            # does conditioning matter at all".
            "zeroed_all": dict(x0_hand=z_ch, touch=z_tc, cond_mask=z_cm,
                               mask_obj=z_mo, mask_hand=z_mh),
        }
        for name, kw in variants.items():
            cond_dict, _ = build_cond(trainer, batch, **kw)
            pred = trainer.models["denoiser"](x_t, t * 1000, cond_dict)
            mse_per = ((pred - target) ** 2).view(B, -1).mean(dim=1)
            stats.add(f"mse/{name}", mse_per.cpu().numpy(), t_bins)

        if (i + 1) % max(1, args.num_batches // 10) == 0:
            print(f"  ... {i + 1}/{args.num_batches} batches")

    ALL = ("baseline", "shuffled_hand", "zeroed_hand",
           "zeroed_image", "zeroed_maskhand", "zeroed_all")
    print_binned(stats, [f"mse/{v}" for v in ALL], "Probe C: mse under conditioning ablations")
    base = stats.mean("mse/baseline", "all")
    print(f"\n  {'variant':<18}{'mse':>12}{'vs baseline':>14}")
    print(f"  {'baseline':<18}{base:>12.6g}{'--':>14}")
    for v in ALL[1:]:
        d = stats.mean(f"mse/{v}", "all")
        print(f"  {v:<18}{d:>12.6g}{100 * (d - base) / base:>13.2f}%")
    h = 100 * (stats.mean("mse/zeroed_hand", "all") - base) / base
    i = 100 * (stats.mean("mse/zeroed_image", "all") - base) / base
    if i > 1e-9:
        print(f"\n  hand / image contribution ratio = {h / i:.3f}  "
              f"(hand {h:.2f}% vs image {i:.2f}%)")
    print(
        "\n  READ: if shuffled_hand ~= baseline, the model is insensitive to WHICH hand it\n"
        "  is given -- confirming finding 1 (no positional embedding on x0h_tokens makes\n"
        "  cross_attn_hand permutation-invariant). If zeroed_hand ~= baseline too, the hand\n"
        "  branch contributes nothing at all. A large gap would FALSIFY finding 1."
    )
    return stats.to_dict()


# --------------------------------------------------------------------------------------
# Probe D -- CFG desync magnitude
# --------------------------------------------------------------------------------------
@torch.no_grad()
def probe_d(trainer, args, p_uncond):
    print("\n" + "=" * 88)
    print(f"PROBE D -- physics losses on CFG-dropped vs kept samples (p_uncond={p_uncond})")
    print("=" * 88)
    trainer.p_uncond = p_uncond
    acc = defaultdict(list)
    it = iter(trainer.dataloader)

    for i in range(args.num_batches):
        try:
            batch = to_cuda(next(it))
        except StopIteration:
            it = iter(trainer.dataloader)
            batch = to_cuda(next(it))

        x_0, x0_hand, touch = batch["x_0"], batch["x0_hand"], batch["touch"]
        B = x_0.shape[0]
        g = torch.Generator(device=x_0.device).manual_seed(args.seed * 15485863 + i)
        t = sample_t_seeded(trainer, B, g, x_0.device)
        noise = torch.randn(x_0.shape, generator=g, device=x_0.device, dtype=x_0.dtype)
        x_t = trainer.diffuse(x_0, t, noise=noise)

        cond_dict, dropped = build_cond(trainer, batch)
        pred = trainer.models["denoiser"](x_t, t * 1000, cond_dict)
        target = trainer.get_v(x_0, noise, t)
        mse_per = ((pred - target) ** 2).view(B, -1).mean(dim=1).cpu().numpy()

        sdf_hand = trainer.ss_dec(x0_hand).float()
        # Exactly what training does: the physics terms use the ORIGINAL, un-dropped
        # x0_hand and touch even for samples whose conditioning was zeroed.
        r = physics_terms(
            trainer.ss_dec, (1.0 - trainer.sigma_min) * noise - pred, sdf_hand, touch
        )
        ni = r.ni_per_sample.cpu().numpy()
        con = r.contact_per_sample.cpu().numpy()

        acc["frac_dropped"].append(dropped.mean())
        for tag, sel in (("dropped", dropped), ("kept", ~dropped)):
            if sel.sum():
                acc[f"mse/{tag}"].extend(mse_per[sel])
                acc[f"ni/{tag}"].extend(ni[sel])
                acc[f"contact/{tag}"].extend(con[sel])

        if (i + 1) % max(1, args.num_batches // 10) == 0:
            print(f"  ... {i + 1}/{args.num_batches} batches")

    out = {}
    print()
    for k in sorted(acc):
        v = np.asarray(acc[k], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "n": int(v.size)}
        print(f"  {k:<20} mean={v.mean():>12.6g}   n={v.size}")
    print(
        "\n  READ: the 'dropped' rows are samples the model could not see the hand for, yet\n"
        "  they are still penalised against the real hand (flow_matching.py:667,697).\n"
        "  A large dropped/kept gap is the size of the bias finding 3 injects into the\n"
        "  unconditional branch that CFG extrapolates away from."
    )
    return out


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher_dir", type=str, required=True,
                    help="Run directory containing config.json and ckpts/")
    ap.add_argument("--ckpt", type=str, default="denoiser_ema0.9999_step0054000.pt",
                    help="Checkpoint filename inside <teacher_dir>/ckpts/")
    ap.add_argument("--data_dir", type=str,
                    default="/projects/gcaddeo/train_flow/TRELLIS/datasets/Leap_Hand",
                    help="Comma-separated dataset roots, same format as train.py")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default outputs/diagnostics/physics_probe_<ckpt>.json)")
    ap.add_argument("--probes", type=str, default="A,B,C,D")
    ap.add_argument("--num_batches", type=int, default=200,
                    help="Batches for probes A, C, D")
    ap.add_argument("--num_batches_grad", type=int, default=5,
                    help="Batches for probe B (each runs 10 t-bins x 5 backwards)")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Default: batch_size_per_gpu from the teacher config")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--p_uncond", type=float, default=0.1, help="Probe D only")
    ap.add_argument("--ni_margin", type=float, default=1.0 / 64.0,
                    help="NI margin in unit-cube SDF units (probes A/B). 0 reproduces "
                         "the un-margined formulation the old teacher trained with; "
                         "the default matches the trainer's fix-5 default.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = edict(json.load(open(os.path.join(args.teacher_dir, "config.json"))))
    ckpt_path = os.path.join(args.teacher_dir, "ckpts", args.ckpt)
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"

    out_path = args.output or os.path.join(
        "outputs", "diagnostics",
        f"physics_probe_{os.path.basename(args.teacher_dir)}_{args.ckpt.replace('.pt', '')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"teacher_dir : {args.teacher_dir}")
    print(f"checkpoint  : {ckpt_path}")
    print(f"data_dir    : {args.data_dir}")
    print(f"output      : {out_path}")

    # --- dataset -------------------------------------------------------------------
    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    print(f"\n{dataset}")

    # --- model + checkpoint --------------------------------------------------------
    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"loaded {len(state)} tensors (strict=True)")

    # The step matters: get_lambda_ni() ramps 50 -> 200 over the first 1000 steps, so a
    # probe left at step 0 would report lambda_ni=50 instead of the 200 used at 54k.
    m = re.search(r"step(\d+)", args.ckpt)
    ckpt_step = int(m.group(1)) if m else 0

    # --- trainer (for encode_image / get_cond / diffuse / get_v / ss_dec) ----------
    trainer_args = dict(cfg.trainer.args)
    if args.batch_size is not None:
        trainer_args["batch_size_per_gpu"] = args.batch_size
    trainer_args["fp16_mode"] = None
    trainer_args["diag_num_workers"] = args.num_workers
    trainer = DiagnosticTrainer(
        {"denoiser": model},
        dataset,
        dataset,
        output_dir=os.path.join(os.path.dirname(out_path), "_trainer_scratch"),
        load_dir=None,      # never call load() -- the checkpoint is already in the model
        step=None,
        **trainer_args,
    )
    trainer.step = ckpt_step
    print(f"\n{trainer}  (step set to {ckpt_step}, lambda_ni={trainer.get_lambda_ni()})")
    print(f"batch_size_per_gpu={trainer.batch_size_per_gpu} "
          f"num_workers={trainer.dataloader.num_workers} "
          f"sigma_min={trainer.sigma_min} t_schedule={trainer.t_schedule}")

    probes = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
    results = {
        "meta": {
            "teacher_dir": args.teacher_dir,
            "ckpt": args.ckpt,
            "data_dir": args.data_dir,
            "num_batches": args.num_batches,
            "num_batches_grad": args.num_batches_grad,
            "batch_size_per_gpu": trainer.batch_size_per_gpu,
            "seed": args.seed,
            "lambda_ni": float(trainer.get_lambda_ni()),
            "lambda_contact": float(trainer.lambda_contact),
            "t_bin_centers": T_BIN_CENTERS,
        }
    }
    if "A" in probes:
        results["probe_a"] = probe_a(trainer, args)
    if "B" in probes:
        results["probe_b"] = probe_b(trainer, args)
    if "C" in probes:
        results["probe_c"] = probe_c(trainer, args)
    if "D" in probes:
        results["probe_d"] = probe_d(trainer, args, args.p_uncond)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
