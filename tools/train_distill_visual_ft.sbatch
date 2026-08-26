#!/bin/bash
#SBATCH --job-name=visual_ft
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# Visual-loss (silhouette presence + space carving) fine-tune of student A
# (EVAL_GUIDANCE.md, user proposal 2026-08-26; correspondence validated by
# tools/validate_mask_column_correspondence.py). Single 24h segment, fresh
# output dir, warm-started from A's 81k EMA — the clean marginal-value
# experiment: compare its final ckpt vs A@81k with the paired a/b.
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2_visual_ft.json"
OUTPUT_DIR="${REPO}/outputs/distill_visual_ft"
INIT_CKPT="${REPO}/outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt"

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

mkdir -p "${OUTPUT_DIR}"
nvidia-smi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ] || { echo "ERROR: DINOv2 not cached" >&2; exit 1; }

EXTRA=""
if ! compgen -G "${OUTPUT_DIR}/ckpts/misc_*.pt" > /dev/null; then
    EXTRA="--load_flow_weights --ckpt_flow_path ${INIT_CKPT}"
    echo "Cold start: warm-starting from ${INIT_CKPT}"
fi

trap 'echo "SIGUSR1: 5 minutes to the wall clock"; kill -TERM ${TRAIN_PID} 2>/dev/null' USR1
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --load_dir "${OUTPUT_DIR}" \
    --ckpt latest \
    ${EXTRA} \
    --num_gpus 2 \
    --auto_retry 3 \
    --data_dir "${DATA_TRAIN}" \
    --data_dir_test "${DATA_TEST}" &
TRAIN_PID=$!
wait ${TRAIN_PID}
RC=$?
echo "=== python exited with ${RC} ==="
exit ${RC}
