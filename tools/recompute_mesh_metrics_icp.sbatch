#!/bin/bash
#SBATCH --job-name=icp_metrics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# ICP-aligned mesh metrics for the saved a/b SDF blobs (CPU-only).
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

python -u tools/recompute_mesh_metrics_icp.py \
    --sdf_dir outputs/diagnostics/ab_guidance_4arm_ycb_sdfs \
    --output  outputs/diagnostics/ab_guidance_4arm_ycb_icp.json
python -u tools/recompute_mesh_metrics_icp.py \
    --sdf_dir outputs/diagnostics/ab_guidance_4arm_sdfs \
    --output  outputs/diagnostics/ab_guidance_4arm_icp.json
