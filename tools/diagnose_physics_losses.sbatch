#!/bin/bash
#SBATCH --job-name=physics_probe
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

# Read-only diagnostics for the teacher's physics losses. Writes only to
# outputs/diagnostics/ -- never to an existing run's ckpts/.
#
#   sbatch tools/diagnose_physics_losses.sbatch
#   sbatch tools/diagnose_physics_losses.sbatch --probes C --num_batches 20   # smoke test
#
# Any extra args are forwarded to the python script.

set -euo pipefail

nvidia-smi

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

# --export=NONE means the interactive environment is gone; source conda explicitly.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper

cd "${REPO}"

# encode_image() lazily torch.hub.load()s DINOv2. Compute nodes may have no egress,
# so fail loudly here rather than hanging inside the probe.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
if [ ! -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]; then
    echo "ERROR: DINOv2 not cached at ${TORCH_HOME}/hub/facebookresearch_dinov2_main" >&2
    echo "Run this once on a login node with network access:" >&2
    echo "  python -c \"import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg',pretrained=True)\"" >&2
    exit 1
fi

python tools/diagnose_physics_losses.py \
    --teacher_dir outputs/flow_conditioned_all_losses_resume_32k_resume3_LEAP \
    --ckpt        denoiser_ema0.9999_step0054000.pt \
    --data_dir    /projects/gcaddeo/train_flow/TRELLIS/datasets/Leap_Hand \
    --num_batches 200 \
    --num_batches_grad 5 \
    "$@"
