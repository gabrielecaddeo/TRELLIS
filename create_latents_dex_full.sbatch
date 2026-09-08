#!/bin/bash
#SBATCH --job-name=dexfull_lat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
# Encode dex-full ss latents (train-repo convention) with the fixed multi-frame script.
# dex-full instances carry ONE frame each (f000; the 8 cameras are 8 instances grouped by
# view_groups.json), hence --frames 1.  Usage:
#   sbatch create_latents_dex_full.sbatch <RANK> <WORLD_SIZE> [INSTANCES_FILE]
set -uo pipefail
RANK="${1:?rank}"; WORLD="${2:?world_size}"; INST="${3:-}"
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"
export PYTHONPATH="${INFER}:${PYTHONPATH:-}"
EXTRA=()
[ -n "${INST}" ] && EXTRA=(--instances_file "${INST}")
echo "=== dex-full latents rank ${RANK}/${WORLD} ${INST:-ALL} $(date) ==="
python -u create_latents_dex_full.py \
    --data_root "${INFER}/dex-full/data_pose_norm" --output_dir "${INFER}/dex-full" \
    --frames 1 --rank "${RANK}" --world_size "${WORLD}" --skip_existing "${EXTRA[@]}"
echo "=== done $(date) ==="
