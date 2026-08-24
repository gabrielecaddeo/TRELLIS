#!/bin/bash
#SBATCH --job-name=dexeval_t2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Paired mesh metrics for the teacher_v2 dex run — flags replicate the historical
# old-teacher run exactly (settings block of summary_ours_ablation_dex_total_total.json).
set -euo pipefail
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"

python -u eval_meshes_paired_emd_voxel_dex.py \
    --gt_dir "${INFER}/models/ycb" \
    --recon_dir "${INFER}/meshes_results_marching_cubes_teacher_v2" \
    --n_points 10000 \
    --seed 0 \
    --cd_squared \
    --pt2tri \
    --normalize bbox_-1_1 \
    --jobs 16 \
    --out_json "${INFER}/summary_teacher_v2_dex_total_total.json" \
    --out_csv "${INFER}/summary_teacher_v2_dex_total_total.csv"
