#!/bin/bash
#SBATCH --job-name=dexcm_dexfull
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
# Canonical ICP dex eval (flags of jobs 475/487, memory trellis-eval-icp) on meshes dumped
# by streaming_eval.py --dump_meshes (layout <arm>/<group>/00/sample.ply; group names start
# with the 3-digit YCB code). Usage: sbatch tools/dex_eval_cm_dexfull.sbatch <RECON_DIR> <TAG>
set -uo pipefail
RECON="${1:?recon dir}"; TAG="${2:?tag}"
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${INFER}"
echo "=== $(date) ICP eval ${TAG} on ${RECON} ($(ls ${RECON} | wc -l) groups)"
python -u eval_meshes_paired_emd_voxel_dex_cm.py \
    --gt_dir "${INFER}/models/ycb" --recon_dir "${RECON}" \
    --cd_squared --normalize bbox_-1_1 --pt2tri --icp --icp_pca --icp_random_restarts 10 \
    --jobs 24 --emd --voxel_iou --voxel_res 64 --voxel_fill --voxel_fixed_bounds --emd_points 512 \
    --out_json "${INFER}/summary_${TAG}_icp.json" --out_csv "${INFER}/summary_${TAG}_icp.csv"
echo "=== done $(date) rc=$?"
