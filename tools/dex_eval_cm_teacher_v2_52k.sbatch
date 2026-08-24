#!/bin/bash
#SBATCH --job-name=dexcm_t2_52k
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Dex-YCB paired metrics for the FROZEN teacher (EMA 52000) meshes, canonical flags
# (user-provided 2026-08-14; identical to job 475). Old-teacher side already exists
# from job 475 (summary_ours_ablation_dex_total_total_icp.json) — meshes unchanged.
set -uo pipefail
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"

python -u eval_meshes_paired_emd_voxel_dex_cm.py \
    --gt_dir "${INFER}/models/ycb" \
    --recon_dir "${INFER}/meshes_results_marching_cubes_teacher_v2_52k" \
    --cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca --icp_random_restarts 10 \
    --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill --voxel_fixed_bounds --emd_points 512 \
    --out_json "${INFER}/summary_teacher_v2_52k_dex_total_total_icp.json" \
    --out_csv  "${INFER}/summary_teacher_v2_52k_dex_total_total_icp.csv"
