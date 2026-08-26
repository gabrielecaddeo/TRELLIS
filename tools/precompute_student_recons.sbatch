#!/bin/bash
#SBATCH --job-name=p4_precompute
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# P4 stage-2 precompute pass (§7.19): frozen best student over the TRAINING
# sets, subsampled views. Resumable — resubmit until "DONE". Shard across the
# two eval-quota-free H200 slots with the two positional args.
#
# Usage: sbatch tools/precompute_student_recons.sbatch [SHARD] [NUM_SHARDS]
# NEEDS USER APPROVAL BEFORE LAUNCH (multi-GPU-day pass).
set -uo pipefail
SHARD="${1:-0}"
NUM_SHARDS="${2:-1}"
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

python -u tools/precompute_student_recons.py \
    --model_dir outputs/distill_teacherv2 \
    --ckpt denoiser_ema0.9999_step0081000.pt \
    --data_dir "${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train" \
    --out_dir "${REPO}/outputs/p4_prior_recons" \
    --shard "${SHARD}" --num_shards "${NUM_SHARDS}"
