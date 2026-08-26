#!/bin/bash
#SBATCH --job-name=p4_smoke
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# P4 smoke test (job-488 pattern): 400 steps, 1 GPU, fresh dir, exercises
# snapshot+save. Verify before the real chain launch:
#  - warm start loads with missing keys == ONLY input_layer_prior.{weight,bias}
#  - step-0 distill_mse ~= A@81k's (~0.05): zero-init prior is a true no-op
#  - prior_keep fraction ~0.7 in logs sane, no NaN, losses in band
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2_p4_recursive_smoke.json"
OUTPUT_DIR="${REPO}/outputs/p4_recursive_smoke"
INIT_CKPT="${REPO}/outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ] || { echo "ERROR: DINOv2 not cached" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --load_dir "${OUTPUT_DIR}" \
    --ckpt latest \
    --load_flow_weights --ckpt_flow_path "${INIT_CKPT}" \
    --num_gpus 1 \
    --data_dir "${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train" \
    --data_dir_test "${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"
