#!/bin/bash
#SBATCH --job-name=stream_dexfull
# dex-full REAL-CAPTURE groups (§7.27): 8 simultaneous cameras/grasp, view v = camera CAMS[v]
# of dex-full-groups/groups_manifest.json. ALWAYS pass --pose_from_hand (no GT inter-camera poses).
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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
python -u tools/streaming_eval.py \
    --data_dir /projects/gcaddeo/inference/TRELLIS/dex-full-groups \
    "$@"
