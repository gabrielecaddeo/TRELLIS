"""Latency table for the ICRA deployment study (ICRA_PLAN item 6, partial:
H200 numbers; rig numbers need the rig).

Measures, batch 1, CUDA-event timed, warmup + N reps:
  - cond encode (DINOv2 + hand/touch conditioning, via get_inference_cond)
  - flow sampling (unguided FlowEulerGuidanceIntervalSampler.sample) for each
    (model x steps) combination
  - SDF decode (ss_dec)
  - marching cubes (CPU, wall clock)
Models: frozen teacher (52k) and any --student_dir/--student_ckpt pairs given.
Output: JSON + printed table under outputs/diagnostics/.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_physics_losses import DiagnosticTrainer, to_cuda  # noqa: E402
from trellis import models, datasets  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402

def cuda_time(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    t = np.array(times)
    return float(t.mean()), float(t.std())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="name:dir:ckpt triplets, e.g. teacher:outputs/teacher_v2_stage2_physics:denoiser_ema0.9999_step0052000.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=[25, 8, 4])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--output", default="outputs/diagnostics/latency_h200.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)

    specs = []
    for m in args.models:
        name, mdir, ckpt = m.split(":")
        specs.append((name, mdir, ckpt))

    # Dataset/trainer scaffolding from the first model's config (cond pipeline is
    # identical across them). Batch size 1 = deployment latency.
    cfg0 = edict(json.load(open(os.path.join(specs[0][1], "config.json"))))
    dataset = getattr(datasets, cfg0.dataset.name)(args.data_dir, **cfg0.dataset.args)
    if hasattr(dataset, "inference"):
        dataset.inference = True
    trainer_args = dict(cfg0.trainer.args)
    trainer_args["batch_size_per_gpu"] = 1
    trainer_args["fp16_mode"] = None
    dummy = getattr(models, cfg0.models.denoiser.name)(**cfg0.models.denoiser.args).cuda()
    trainer = DiagnosticTrainer({"denoiser": dummy}, dataset, dataset,
                                output_dir="outputs/diagnostics/_trainer_scratch_lat",
                                load_dir=None, step=None, **trainer_args)
    data = to_cuda(next(iter(trainer.dataloader)))
    x_0 = data.pop("x_0")

    res = {"meta": {"reps": args.reps, "batch": 1, "steps": args.steps,
                    "gpu": torch.cuda.get_device_name(0)}}

    mean, std = cuda_time(lambda: trainer.get_inference_cond(**data), reps=args.reps)
    res["cond_encode_ms"] = {"mean": mean, "std": std}
    print(f"cond encode: {mean:.1f} +- {std:.1f} ms")

    cond_args = trainer.get_inference_cond(**data)
    pos, neg = cond_args["cond"], cond_args["neg_cond"]
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=trainer.sigma_min)
    noise = torch.randn_like(x_0)

    del dummy
    torch.cuda.empty_cache()

    latent_holder = {}
    for name, mdir, ckpt in specs:
        cfg = edict(json.load(open(os.path.join(mdir, "config.json"))))
        model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
        sd = torch.load(os.path.join(mdir, "ckpts", ckpt), map_location="cuda", weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        res[name] = {"ckpt": ckpt, "params_M": n_params / 1e6}
        for steps in args.steps:
            def run():
                with torch.no_grad():
                    out = sampler.sample(model, noise, cond=pos, neg_cond=neg,
                                         steps=steps, rescale_t=3.0, cfg_strength=5.0,
                                         cfg_interval=(0.5, 1.0), verbose=False)
                latent_holder["z"] = out.samples
            mean, std = cuda_time(run, reps=args.reps)
            res[name][f"flow_{steps}steps_ms"] = {"mean": mean, "std": std}
            print(f"{name} ({n_params/1e6:.0f}M) flow {steps:>2} steps: {mean:7.1f} +- {std:.1f} ms")
        del model
        torch.cuda.empty_cache()

    z = latent_holder["z"]
    mean, std = cuda_time(lambda: trainer.ss_dec(z), reps=args.reps)
    res["decode_ms"] = {"mean": mean, "std": std}
    print(f"ss_dec decode: {mean:.1f} +- {std:.1f} ms")

    with torch.no_grad():
        sdf = trainer.ss_dec(z).float()[0, 0].cpu().numpy()
    from skimage.measure import marching_cubes
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter(); marching_cubes(np.transpose(sdf, (2, 1, 0)), level=0.0); ts.append((time.perf_counter() - t0) * 1e3)
    res["marching_cubes_ms"] = {"mean": float(np.mean(ts)), "std": float(np.std(ts))}
    print(f"marching cubes (CPU): {np.mean(ts):.1f} +- {np.std(ts):.1f} ms")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
