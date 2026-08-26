#!/bin/bash
#SBATCH --job-name=dexcm_student
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Canonical ICP eval for the student dex meshes (pair of dex_student.sbatch).
# Flags identical to jobs 475/487 (memory trellis-eval-icp).
#
# Usage: sbatch --dependency=afterany:<gpu_job> tools/dex_eval_cm_student.sbatch <TAG> <STEPS>
set -uo pipefail
TAG="${1:?usage: dex_eval_cm_student.sbatch <TAG> <STEPS>}"
STEPS="${2:?missing STEPS}"
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"

python -u eval_meshes_paired_emd_voxel_dex_cm.py \
    --gt_dir "${INFER}/models/ycb" \
    --recon_dir "${INFER}/meshes_results_marching_cubes_student_${TAG}_s${STEPS}" \
    --cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca --icp_random_restarts 10 \
    --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill --voxel_fixed_bounds --emd_points 512 \
    --out_json "${INFER}/summary_student_${TAG}_s${STEPS}_dex_total_total_icp.json" \
    --out_csv  "${INFER}/summary_student_${TAG}_s${STEPS}_dex_total_total_icp.csv"
