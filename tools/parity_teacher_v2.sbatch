#!/bin/bash
#SBATCH --job-name=parity_t2
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# teacher_v2 deployment parity: convert ckpt -> safetensors, generate train-repo
# reference, verify the ported inference repo reproduces it bit-for-bit(ish).
set -euo pipefail
TRAIN=/projects/gcaddeo/train_flow_conditioned/TRELLIS
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
CKPT=${PARITY_CKPT:-$(ls -1 ${TRAIN}/outputs/teacher_v2_stage2_physics/ckpts/denoiser_ema0.9999_step*.pt | sort | tail -1)}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Parity checkpoint: ${CKPT}"

# 1. convert to the deployed safetensors name
python -u "${INFER}/teacher_v2_port/convert_teacher_v2.py" --pt "${CKPT}"

# 2. train-side reference
cd "${TRAIN}"
PYTHONPATH="${TRAIN}" python -u tools/parity_teacher_v2_ref.py \
    --ckpt "${CKPT}" \
    --decoder_st "${INFER}/teacher_v2_stage2/ckpts/decoder_ema0.9999_step0300000.safetensors" \
    --out "${INFER}/teacher_v2_port/parity_ref.pt"

# 3. inference-side check
cd "${INFER}"
PYTHONPATH="${INFER}" python -u teacher_v2_port/parity_check.py \
    --ref "${INFER}/teacher_v2_port/parity_ref.pt"
