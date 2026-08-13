"""Inference-repo half of the teacher_v2 deployment parity test.

Loads the converted denoiser + deployed decoder through the inference repo's own
loading path (models.from_pretrained on the teacher_v2_stage2 pipeline ckpts), then
recomputes exactly what tools/parity_teacher_v2_ref.py produced in the train repo:
same synthetic inputs, same three raw velocities, same guided 25-step rollout.

Prints PARITY PASS / PARITY FAIL; exit code follows.
Run under PYTHONPATH=<inference repo>. Same determinism settings as the reference.
"""
import argparse
import sys

import torch


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
    ap.add_argument("--ref", required=True, help="parity reference .pt from the train repo")
    ap.add_argument("--ckpts", default="/projects/gcaddeo/inference/TRELLIS/teacher_v2_stage2/ckpts")
    ap.add_argument("--tol_v", type=float, default=1e-4)
    ap.add_argument("--tol_sdf", type=float, default=1e-3)
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
    ref = torch.load(args.ref, map_location="cpu")

    model = models.from_pretrained(f"{args.ckpts}/denoiser_teacher_v2_final").to(dev).eval()
    decoder = models.from_pretrained(f"{args.ckpts}/decoder_ema0.9999_step0300000").to(dev).eval()

    ok = True

    dec_fp = {k: float(v.double().sum()) for k, v in sorted(decoder.state_dict().items())}
    fp_diff = max(
        abs(dec_fp.get(k, float("nan")) - v) for k, v in ref["decoder_fingerprint"].items()
    )
    same_keys = set(dec_fp) == set(ref["decoder_fingerprint"])
    print(f"decoder fingerprint: keys match={same_keys}, max |sum diff|={fp_diff:.3e}")
    if not same_keys or fp_diff > 1e-3:
        ok = False

    cond, neg, x_t, noise = make_inputs(dev)

    es = FlowEulerSampler(sigma_min=1e-5)
    with torch.no_grad():
        for t, v_ref in ref["v_at_t"].items():
            v = es._inference_model(model, x_t, t, cond).float().cpu()
            d = (v - v_ref).abs().max().item()
            print(f"velocity parity @ t={t}: max|diff|={d:.3e} (tol {args.tol_v})")
            if d > args.tol_v:
                ok = False

    gis = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    res = gis.sample_velocity_conditioned(
        model, noise.clone(), decoder, cond, neg_cond=neg,
        steps=25, rescale_t=3.0, cfg_strength=5.0, cfg_interval=(0.5, 1.0),
        alpha_vel=10.0, verbose=False,
    )
    z = res.samples.detach().float().cpu()
    with torch.no_grad():
        sdf = decoder(res.samples.detach()).float().cpu()
    dz = (z - ref["z_rollout"]).abs().max().item()
    dsdf = (sdf - ref["sdf_rollout"]).abs().max().item()
    print(f"rollout latent parity: max|diff|={dz:.3e}")
    print(f"rollout SDF parity:    max|diff|={dsdf:.3e} (tol {args.tol_sdf})")
    if dsdf > args.tol_sdf:
        ok = False

    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
