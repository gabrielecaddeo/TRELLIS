"""Generate the train-repo reference for the teacher_v2 deployment parity test.

Loads the teacher_v2 denoiser from its train checkpoint and the *deployed* SS decoder
weights (the inference repo's safetensors, through the train repo's identical class),
then computes on fixed synthetic inputs:
  (a) raw model velocities at three t values,
  (b) a full 25-step sample_velocity_conditioned rollout (deployment params) + decoded SDF.
The inference-side teacher_v2_port/parity_check.py must reproduce all of it.

Run under PYTHONPATH=<train repo>. Deterministic: TF32 off, cudnn deterministic.
"""
import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file


def make_inputs(device):
    g = torch.Generator(device="cpu").manual_seed(1234)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    def rand(*shape):
        return torch.rand(*shape, generator=g).to(device)

    cond = {
        "cond": randn(1, 1374, 1024),
        "cond_mask": randn(1, 1374, 1024),
        "mask_hand": rand(1, 37, 37),
        "mask_obj": rand(1, 37, 37),
        "x0_hand": randn(1, 8, 16, 16, 16),
        "touch": (rand(1, 2, 64, 64, 64) < 0.02).float(),
    }
    neg = {k: torch.zeros_like(v) for k, v in cond.items()}
    x_t = randn(1, 8, 16, 16, 16)
    noise = randn(1, 8, 16, 16, 16)
    return cond, neg, x_t, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="teacher_v2 denoiser EMA .pt")
    ap.add_argument("--decoder_st", required=True, help="deployed SS decoder .safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import trellis.models as models
    from trellis.pipelines.samplers.flow_euler import (
        FlowEulerSampler,
        FlowEulerGuidanceIntervalSampler,
    )

    dev = "cuda"

    dcfg = json.load(open("outputs/teacher_v2_stage2_physics/config.json"))["models"]["denoiser"]
    dargs = dict(dcfg["args"])
    dargs["use_checkpoint"] = False
    model = getattr(models, dcfg["name"])(**dargs)
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model = model.to(dev).eval()

    vcfg = json.load(open("/projects/gcaddeo/train_flow/TRELLIS/outputs/vae_final_all_resume_2/config.json"))["models"]["decoder"]
    decoder = getattr(models, vcfg["name"])(**vcfg["args"])
    dec_sd = load_file(args.decoder_st)
    decoder.load_state_dict(dec_sd)
    decoder = decoder.to(dev).eval()
    dec_fp = {k: float(v.double().sum()) for k, v in sorted(dec_sd.items())}

    cond, neg, x_t, noise = make_inputs(dev)

    es = FlowEulerSampler(sigma_min=1e-5)
    v_at_t = {}
    with torch.no_grad():
        for t in (1.0, 0.55, 0.1):
            v_at_t[t] = es._inference_model(model, x_t, t, cond).float().cpu()

    gis = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    res = gis.sample_velocity_conditioned(
        model, noise.clone(), decoder, cond, neg_cond=neg,
        steps=25, rescale_t=3.0, cfg_strength=5.0, cfg_interval=(0.5, 1.0),
        alpha_vel=10.0, verbose=False,
    )
    z = res.samples.detach()
    with torch.no_grad():
        sdf = decoder(z).float().cpu()

    torch.save(
        {
            "v_at_t": v_at_t,
            "z_rollout": z.float().cpu(),
            "sdf_rollout": sdf,
            "decoder_fingerprint": dec_fp,
            "ckpt": args.ckpt,
        },
        args.out,
    )
    print(f"Reference written to {args.out}")
    print(f"  rollout latent norm {z.norm().item():.6f}, sdf mean {sdf.mean().item():.6f}")


if __name__ == "__main__":
    main()
