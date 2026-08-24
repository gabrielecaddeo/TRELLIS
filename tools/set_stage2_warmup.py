"""
Fix the lambda_ni warmup for stage 2 (TEACHER_RETRAIN.md §5, "the lambda warmup is a no-op
on resume").

`FlowMatchingTrainer.get_lambda_ni()` compares `self.step` against
`lambda_non_interpenetration_warmup`. On resume `self.step` continues from stage 1's final
step, which is far past the config's 1000, so lambda jumps straight to its maximum on the
very first stage-2 step -- exactly what bit the previous lineage, where every `all_losses`
run effectively trained at lambda=200 from step one.

This rewrites the stage-2 config so the warmup ends `--ramp` steps after stage 1 stopped.

    python tools/set_stage2_warmup.py \
        --stage1_dir outputs/teacher_v2_stage1_cond \
        --config configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage2.json \
        --ramp 2000
"""
import argparse
import collections
import glob
import json
import os


def latest_step(stage1_dir: str) -> int:
    files = glob.glob(os.path.join(stage1_dir, 'ckpts', 'misc_*.pt'))
    if not files:
        raise SystemExit(f'no misc_*.pt checkpoints under {stage1_dir}/ckpts')
    return max(int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1_dir', default='outputs/teacher_v2_stage1_cond')
    ap.add_argument('--config',
                    default='configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage2.json')
    ap.add_argument('--ramp', type=int, default=2000,
                    help='steps over which lambda_ni ramps from start to max')
    args = ap.parse_args()

    step = latest_step(args.stage1_dir)
    cfg = json.load(open(args.config), object_pairs_hook=collections.OrderedDict)
    warmup = step + args.ramp
    cfg['trainer']['args']['lambda_non_interpenetration_warmup'] = warmup
    json.dump(cfg, open(args.config, 'w'), indent=4)

    a = cfg['trainer']['args']
    print(f'stage 1 last checkpoint: step {step}')
    print(f'lambda_non_interpenetration_warmup -> {warmup} '
          f'({a["lambda_non_interpenetration_start"]} -> {a["lambda_non_interpenetration_max"]} '
          f'over {args.ramp} steps)')


if __name__ == '__main__':
    main()
