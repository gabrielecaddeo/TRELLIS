#!/bin/bash
#SBATCH --job-name=distill_smoke
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# GPU smoke test for the FIXED distillation trainer (ICRA_PLAN.md item 2).
# Runs the *_distill_teacherv2_smoke.json config: identical to the real
# distillation config except max_steps 400, i_log 50 (log.txt only flushes on
# step % i_log == 0), i_sample 200 and i_save 300 so the snapshot and
# checkpoint-save paths are exercised before any 24h launch depends on them.
#
# What to verify in the output (ICRA_PLAN.md / EVAL_GUIDANCE.md §3):
#   - "Loaded distillation teacher" with the FROZEN ckpt path, no missing/
#     unexpected keys (strict load)
#   - ni_floor / contact_floor keys in log.txt lines
#   - loss magnitudes sane, distill_mse decreasing, no NaN
#
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2_smoke.json"
OUTPUT_DIR="${REPO}/outputs/distill_teacherv2_smoke"

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

nvidia-smi

# --export=NONE means the interactive environment is gone; source conda explicitly.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
if [ ! -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]; then
    echo "ERROR: DINOv2 not cached at ${TORCH_HOME}/hub/facebookresearch_dinov2_main" >&2
    exit 1
fi

echo "=== distill smoke job=${SLURM_JOB_ID} node=$(hostname) ==="
echo "config=${CONFIG}"
echo "output=${OUTPUT_DIR}"

# --auto_retry 0: a smoke test should fail fast and visibly, not mask a crash.
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --load_dir "${OUTPUT_DIR}" \
    --ckpt latest \
    --num_gpus 1 \
    --auto_retry 0 \
    --data_dir "${DATA_TRAIN}" \
    --data_dir_test "${DATA_TEST}"
RC=$?
echo "=== python exited with ${RC} ==="
exit ${RC}
