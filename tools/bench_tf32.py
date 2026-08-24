"""
Measure what the fp32 / TF32 / determinism settings actually cost, on the real denoiser.

train.py runs with torch defaults, which since PyTorch 1.12 means
`torch.backends.cuda.matmul.allow_tf32 = False` -- every fp32 matmul takes the non-tensor-core
path. On an H200 that is ~67 TFLOPS instead of ~495 TFLOPS dense TF32. It also sets
`use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:16:8`, which
restricts cuBLAS kernel choice -- while the run is not actually deterministic anyway, because
the memory-efficient attention backward warns and stays non-deterministic.

This times a real forward+backward under each combination so the decision is made on measured
numbers rather than spec sheets.

    python tools/bench_tf32.py --batch 16 --iters 8
"""
import argparse
import json
import os
import time

import torch

from trellis import models


def build(cfg_path, device='cuda'):
    cfg = json.load(open(cfg_path))
    m = getattr(models, cfg['models']['denoiser']['name'])(**cfg['models']['denoiser']['args'])
    return m.to(device).train(), cfg


def make_batch(m, B, device):
    R = m.resolution
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(B, m.in_channels, R, R, R, device=device, generator=g)
    t = torch.full((B,), 500.0, device=device)
    cond = {
        'cond':      torch.randn(B, 1374, m.cond_channels, device=device, generator=g),
        'cond_mask': torch.randn(B, 1374, m.cond_channels, device=device, generator=g),
        'mask_hand': torch.rand(B, 37, 37, device=device, generator=g),
        'mask_obj':  torch.rand(B, 37, 37, device=device, generator=g),
        'x0_hand':   torch.randn(B, m.in_channels, R, R, R, device=device, generator=g),
        'touch':     torch.rand(B, 2, 64, 64, 64, device=device, generator=g),
    }
    target = torch.randn_like(x)
    return x, t, cond, target


def timeit(m, batch, iters, warmup=2):
    x, t, cond, target = batch
    opt = torch.optim.AdamW(m.parameters(), lr=1e-9)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(m(x, t, dict(cond)), target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def sweep(args):
    """Is the VRAM being used well? Measure batch x gradient-checkpointing, all with TF32 on.

    Reports throughput in SAMPLES/s, not steps/s -- a bigger batch trivially lowers steps/h
    while doing the same or more work, so steps/h is the wrong axis for this comparison.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = json.load(open(args.config))
    rows = []
    for ckpt in (True, False):
        for B in [int(b) for b in args.sweep_batches.split(',')]:
            margs = dict(cfg['models']['denoiser']['args'])
            margs['use_checkpoint'] = ckpt
            tag = f'batch {B:>3}/GPU  checkpoint={str(ckpt):<5}'
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m = getattr(models, cfg['models']['denoiser']['name'])(**margs).cuda().train()
                t = timeit(m, make_batch(m, B, 'cuda'), args.iters, warmup=2)
                mem = torch.cuda.max_memory_allocated() / 2**30
                rows.append((tag, t, B / t, mem))
                print(f'{tag}  {t:6.2f} s/step  {B / t:6.2f} samp/s  {mem:6.1f} GiB')
            except torch.cuda.OutOfMemoryError:
                rows.append((tag, None, None, None))
                print(f'{tag}     OOM')
            finally:
                del m
                torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f'\n{"config":<34}{"s/step":>9}{"samples/s":>11}{"peak GiB":>10}{"VRAM used":>11}')
    for tag, t, sps, mem in rows:
        if t is None:
            print(f'{tag:<34}{"OOM":>9}')
        else:
            print(f'{tag:<34}{t:>9.2f}{sps:>11.2f}{mem:>10.1f}{100 * mem / total:>10.0f}%')
    print(f'\nGPU total: {total:.0f} GiB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',
                    default='configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--sweep', action='store_true',
                    help='sweep batch x use_checkpoint instead of the TF32 comparison')
    ap.add_argument('--sweep_batches', type=str, default='16,32,48')
    args = ap.parse_args()

    if args.sweep:
        return sweep(args)

    m, _ = build(args.config)
    batch = make_batch(m, args.batch, 'cuda')
    print(f'denoiser fwd+bwd+step, batch={args.batch}/GPU, {args.iters} iters\n')

    results = {}

    # Exactly what train.py does today.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    results['as_configured (fp32, deterministic)'] = timeit(m, batch, args.iters)

    # Same, but let fp32 matmuls use the tensor cores. TF32 keeps the fp32 exponent range and
    # accumulates in fp32; only the matmul input mantissa drops 23 -> 10 bits.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results['+ TF32 matmuls'] = timeit(m, batch, args.iters)

    # ...and drop the determinism constraint, which is not delivering determinism anyway
    # (memory-efficient attention backward is non-deterministic and only warns).
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    results['+ TF32 + non-deterministic'] = timeit(m, batch, args.iters)

    base = results['as_configured (fp32, deterministic)']
    print(f"{'setting':<38}{'s/step':>9}{'speedup':>9}{'steps/h':>9}{'72h steps':>11}")
    for k, v in results.items():
        print(f'{k:<38}{v:>9.2f}{base / v:>8.2f}x{3600 / v:>9.0f}{72 * 3600 / v:>11.0f}')
    print(f'\npeak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB '
          f'of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB')


if __name__ == '__main__':
    main()
