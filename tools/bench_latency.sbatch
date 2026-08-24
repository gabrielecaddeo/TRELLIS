#!/bin/bash
#SBATCH --job-name=bench_lat
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
python -u tools/bench_latency.py \
    --data_dir "${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test" \
    --models teacher:outputs/teacher_v2_stage2_physics:denoiser_ema0.9999_step0052000.pt \
             student8x2:outputs/distill_teacherv2:denoiser_ema0.9999_step0048000.pt \
    "$@"
