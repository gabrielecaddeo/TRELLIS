"""
The stage-1 -> stage-2 gate: is the hand branch actually carrying signal yet?

TEACHER_RETRAIN.md §5 says to run stage 1 "until mse plateaus", but at the measured ~240
steps/h the 72-hour budget buys ~17k steps and the old lineage needed ~44k to plateau. So
"plateau" would never fire and the stage boundary would be set by the clock rather than by
evidence. This replaces that with the measurement that actually gates physics.

Why this is the right gate: `cross_attn_hand.to_out` is zero-initialised, so at step 0 the hand
branch contributes exactly nothing to the output (verified: max|out(hand) - out(no hand)| = 0).
Physics losses applied then are scored against a hand the network structurally cannot see --
the same impossible objective probe D measured for CFG-dropped samples, but for 100% of the
batch. Physics becomes meaningful only once permuting/removing the hand actually changes the
prediction. Probe C measures exactly that, in ~6 minutes.

Usage (normally via tools/check_hand_gate.sbatch, which runs the probe first):

    python tools/check_hand_gate.py --run_dir outputs/teacher_v2_stage1_cond           # verdict only
    python tools/check_hand_gate.py --run_dir ... --latest_ckpt                        # print ckpt name

Verdicts are a RATIO to the image pathway's own contribution, measured in the same probe run
(`mse/zeroed_image`), not an absolute mse percentage:

    < 0.10   NOT READY   -- the hand barely registers; keep training stage 1
    0.10-0.25 EMERGING   -- live and growing; keep training
    > 0.25   READY       -- switch to stage 2

Why a ratio. The first version used absolute thresholds (5% / 15%) chosen only as "clearly
above the old teacher's 2%" -- arbitrary, and sensitive to drift in baseline mse. Measuring the
image ablation gave a real denominator: at step 4000 the image pathway is worth +19.27% of mse
while the hand is worth +2.52%. The hand SHOULD be worth less -- the image determines what the
object is, the hand only constrains how it is held -- so the question is not "is the hand big"
but "is it a meaningful fraction of the dominant pathway".

Context for reading the numbers:
  - old teacher @ step 54000 : shuffled +1.99%, zeroed +1.03%  (a model ignoring the hand)
  - teacher_v2  @ step  2000 : shuffled +1.04%, zeroed +0.64%
  - teacher_v2  @ step  4000 : shuffled +2.52%, zeroed +1.42%, image +19.27% -> ratio 0.131
  - `zeroed_maskhand` measured +0.00% at step 4000: the 2-D hand-mask branch contributes
    nothing, presumably redundant against the far richer 3-D `x0_hand`.
"""
import argparse
import glob
import json
import os
import re


def latest_ckpt(run_dir: str, kind: str = 'raw') -> str:
    """Newest checkpoint by step number.

    kind='raw'  -> denoiser_step*.pt      : the LIVE training weights
    kind='ema'  -> denoiser_ema*_step*.pt : the deployable weights

    Default is 'raw', and that matters. With ema_rate=0.9999 the EMA is still 82% initial
    weights at step 2000, and `cross_attn_hand.to_out` is zero-initialised (finding I), so the
    EMA's copy of the hand branch is ~10x weaker than the live one early on. Gating on the EMA
    at step 2000 measured EMA lag, not the model -- it returned +0.00% while the live weights
    had already grown the branch from exactly 0 to ||W||=1.21. The live weights are also the
    ones that will receive stage-2's physics gradients, so they are what the gate should read.
    Check the EMA separately at the END of stage 1, since that is what gets deployed.
    """
    pat = 'denoiser_ema*_step*.pt' if kind == 'ema' else 'denoiser_step*.pt'
    files = [f for f in glob.glob(os.path.join(run_dir, 'ckpts', pat))
             if kind == 'ema' or 'ema' not in os.path.basename(f)]
    if not files:
        raise SystemExit(f'no {pat} under {run_dir}/ckpts '
                         f'(i_save=2000, so none exists before step 2000)')
    def step_of(f):
        m = re.search(r'step(\d+)\.pt$', os.path.basename(f))
        return int(m.group(1)) if m else -1
    return os.path.basename(max(files, key=step_of))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='outputs/teacher_v2_stage1_cond')
    ap.add_argument('--ckpt', default=None, help='default: newest checkpoint of --weights kind')
    ap.add_argument('--weights', choices=['raw', 'ema'], default='raw',
                    help="'raw' = live training weights (default; EMA lags badly before "
                         "~10k steps). 'ema' = deployable weights, check at end of stage 1.")
    ap.add_argument('--latest_ckpt', action='store_true',
                    help='just print the resolved checkpoint filename and exit')
    ap.add_argument('--json', default=None, help='override the probe JSON path')
    # Thresholds are RATIOS to the image pathway's own contribution, measured in the same
    # probe run, not absolute mse percentages. An absolute number would be arbitrary (the
    # first version used 15%, picked only as "clearly above the old teacher's 2%") and would
    # also drift as the model's baseline mse changes. Anchoring to `zeroed_image` makes the
    # gate self-calibrating: it asks "is the hand worth a meaningful fraction of what the
    # image is worth?" -- and the hand SHOULD be worth less, since the image determines what
    # the object is while the hand only constrains how it is held.
    ap.add_argument('--ready_ratio', type=float, default=0.25,
                    help='hand effect / image effect to call READY (default 0.25)')
    ap.add_argument('--emerging_ratio', type=float, default=0.10)
    args = ap.parse_args()

    ckpt = args.ckpt or latest_ckpt(args.run_dir, args.weights)
    if args.latest_ckpt:
        print(ckpt)
        return 0

    # Dedicated path: diagnose_physics_losses.py writes one JSON per (run, ckpt) and a probe
    # subset overwrites the whole file, so a bare --probes C run would clobber A/B/D results.
    # (That is exactly what happened on 2026-08-06: job 411's A/B/D overwrote job 410's C.)
    jpath = args.json or os.path.join(
        'outputs', 'diagnostics',
        f'handgate_{os.path.basename(args.run_dir.rstrip("/"))}_{ckpt.replace(".pt", "")}.json')
    if not os.path.exists(jpath):
        raise SystemExit(f'no probe output at {jpath} -- run tools/check_hand_gate.sbatch first')

    blob = json.load(open(jpath))
    if 'probe_c' not in blob:
        raise SystemExit(f'{jpath} has no probe_c (contains: {sorted(blob)}). '
                         f'It was written by a run that did not include --probes C.')
    c = blob['probe_c']
    base = c['mse/baseline']['all']
    shuf = c['mse/shuffled_hand']['all']
    zero = c['mse/zeroed_hand']['all']
    d_shuf = 100 * (shuf - base) / base
    d_zero = 100 * (zero - base) / base

    step = int(re.search(r'step(\d+)', ckpt).group(1))
    print(f'checkpoint          : {ckpt}  (step {step})')
    if 'ema' in ckpt and step < 10000:
        frac = 0.9999 ** step
        print(f'  !! EMA WARNING: at step {step} the EMA is {100 * frac:.0f}% initial weights. '
              f'Since the hand\n     branch starts at exactly zero, this reads ~0 regardless of '
              f'training. Use --weights raw.')
    print(f'mse/baseline        : {base:.5f}')
    print(f'mse/shuffled_hand   : {shuf:.5f}   ({d_shuf:+.2f}% vs baseline)')
    print(f'mse/zeroed_hand     : {zero:.5f}   ({d_zero:+.2f}% vs baseline)')
    print(f'  old broken teacher: +1.99% shuffled / +1.03% zeroed')

    # shuffled_hand is the primary signal: physics pushes the object to avoid/contact THIS
    # hand, so what matters is whether the model responds to the specific hand it is given.
    # zeroed_hand only asks whether a hand helps at all.
    key = max(d_shuf, d_zero)

    img = c.get('mse/zeroed_image')
    if img is None:
        print('\n  (no mse/zeroed_image in this probe run -- falling back to absolute '
              'thresholds; re-run the gate to get the image-anchored verdict)')
        ready, emerging, scale, unit = 15.0, 5.0, key, '%'
    else:
        d_img = 100 * (img['all'] - base) / base
        print(f'mse/zeroed_image    : {img["all"]:.5f}   ({d_img:+.2f}% vs baseline)  <- reference')
        if d_img <= 1e-6:
            raise SystemExit('image ablation shows no effect; cannot anchor the gate')
        scale = key / d_img
        ready, emerging, unit = args.ready_ratio, args.emerging_ratio, ' of image'
        print(f'\nhand / image ratio  : {scale:.3f}   '
              f'(hand {key:+.2f}% vs image {d_img:+.2f}%)')

    if scale >= ready:
        verdict, action = 'READY', 'switch to stage 2 (see §9 CP5)'
    elif scale >= emerging:
        verdict, action = 'EMERGING', 'branch is live and growing -- keep training stage 1'
    else:
        verdict, action = 'NOT READY', 'hand barely registers -- keep training stage 1'
    shown = f'{scale:.3f}{unit}' if unit != '%' else f'{scale:+.2f}%'
    print(f'GATE: {verdict}  ({shown}; ready at {ready}{unit})  ->  {action}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
