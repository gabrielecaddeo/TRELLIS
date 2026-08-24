#!/bin/bash
#SBATCH --job-name=ab_guidance
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# A/B evaluation of physics guidance at sampling time (tools/ab_eval_guidance.py).
# Runs entirely in the train repo on the held-out split. Read-only apart from
# outputs/diagnostics/. Extra args are forwarded to the python script.
#
#   sbatch tools/ab_eval_guidance.sbatch                     # latest EMA ckpt, defaults
#   sbatch tools/ab_eval_guidance.sbatch --alpha_vel 50      # e.g. stronger guidance
#
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

python -u tools/ab_eval_guidance.py \
    --data_dir "${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test" \
    "$@"
