#!/bin/bash
#SBATCH --job-name=dex_student
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Dex-YCB inference for a DISTILLED STUDENT: converts the named train-repo EMA
# ckpt into a fresh student pipeline dir (generalized convert_teacher_v2.py:
# arch json from the train config, decoders symlinked from the teacher), then
# runs the same inference procedure as jobs 486/494-497 (seed 42, view 0,
# resumable). Pair with tools/dex_eval_cm_student.sbatch <TAG> <STEPS> via
# --dependency=afterany for the canonical ICP eval (487 pattern).
#
# Usage: sbatch tools/dex_student.sbatch <TRAIN_DIR> <CKPT_FILE> <TAG> <STEPS>
#   e.g. sbatch tools/dex_student.sbatch \
#          /projects/gcaddeo/train_flow_conditioned/TRELLIS/outputs/distill_teacherv2 \
#          denoiser_ema0.9999_step0081000.pt a81k 8
set -euo pipefail
TRAIN_DIR="${1:?usage: dex_student.sbatch <TRAIN_DIR> <CKPT_FILE> <TAG> <STEPS>}"
CKPT="${2:?missing CKPT_FILE}"
TAG="${3:?missing TAG}"
STEPS="${4:?missing STEPS}"
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${INFER}"
PIPE="${INFER}/student_${TAG}_stage2"
echo "=== convert ${TRAIN_DIR}/ckpts/${CKPT} -> ${PIPE} ==="
PYTHONPATH="${INFER}" python -u teacher_v2_port/convert_teacher_v2.py \
    --pt "${TRAIN_DIR}/ckpts/${CKPT}" \
    --train_config "${TRAIN_DIR}/config.json" \
    --pipeline_dir "${PIPE}" \
    --name "denoiser_student_${TAG}_final"

echo "=== dex inference, student ${TAG}, steps=${STEPS} ==="
PYTHONPATH="${INFER}" \
DEX_PIPELINE_DIR="${PIPE}" \
DEX_STEPS="${STEPS}" \
DEX_OUT_ROOT="${INFER}/meshes_results_marching_cubes_student_${TAG}_s${STEPS}" \
python -u teacher_v2_port/inference_dex_teacher_v2.py
