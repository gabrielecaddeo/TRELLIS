#!/bin/bash
#SBATCH --job-name=dexcm_t2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Dex-YCB paired metrics with the *_cm* script — the variant that produced the
# historical summary_ours_ablation_dex_total_total.json (keys CD_cm2/CD_sqrt_mm/
# Fscore5/Fscore10 prove the provenance). ICP ALWAYS ON (user directive 2026-08-14):
# both sides evaluated with identical ICP flags so the comparison is pose-agnostic.
#   1. old-teacher meshes -> summary_ours_ablation_dex_total_total_icp.json
#   2. teacher_v2 meshes  -> summary_teacher_v2_dex_total_total_icp.json
set -uo pipefail
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"

# Exact historical invocation flags (user-provided 2026-08-14):
COMMON="--gt_dir ${INFER}/models/ycb --cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca --icp_random_restarts 10 --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill --voxel_fixed_bounds --emd_points 512"

echo "=== old-teacher meshes (ICP) ==="
python -u eval_meshes_paired_emd_voxel_dex_cm.py ${COMMON} \
    --recon_dir "${INFER}/meshes_results_marching_cubes_ablation_dex" \
    --out_json "${INFER}/summary_ours_ablation_dex_total_total_icp.json" \
    --out_csv  "${INFER}/summary_ours_ablation_dex_total_total_icp.csv"

echo "=== teacher_v2 meshes (ICP) ==="
python -u eval_meshes_paired_emd_voxel_dex_cm.py ${COMMON} \
    --recon_dir "${INFER}/meshes_results_marching_cubes_teacher_v2" \
    --out_json "${INFER}/summary_teacher_v2_dex_total_total_icp.json" \
    --out_csv  "${INFER}/summary_teacher_v2_dex_total_total_icp.csv"
