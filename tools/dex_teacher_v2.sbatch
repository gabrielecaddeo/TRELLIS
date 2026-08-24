#!/bin/bash
#SBATCH --job-name=dex_t2
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Dex-YCB benchmark with teacher_v2: convert the FINAL EMA checkpoint (latest on
# disk at run time, i.e. after job 448 ends), then run the 994-instance inference.
# Resumable: resubmitting continues where the wall cut it off.
set -euo pipefail
TRAIN=/projects/gcaddeo/train_flow_conditioned/TRELLIS
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
CKPT="${DEX_CKPT:-$(ls -1 ${TRAIN}/outputs/teacher_v2_stage2_physics/ckpts/denoiser_ema0.9999_step*.pt | sort | tail -1)}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Final teacher checkpoint: ${CKPT}"
python -u "${INFER}/teacher_v2_port/convert_teacher_v2.py" --pt "${CKPT}"

cd "${INFER}"
PYTHONPATH="${INFER}" DEX_OUT_ROOT="${DEX_OUT_ROOT:-}" python -u teacher_v2_port/inference_dex_teacher_v2.py
