#!/bin/bash
#SBATCH --job-name=dex_t2_steps
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Dex-YCB inference with the FROZEN teacher_v2 (EMA 52000) at a reduced step
# count, for the steps axis of the Pareto table (EVAL_GUIDANCE.md §7.1 showed
# step-invariance on held-out synthetic; this is the real-capture confirmation).
#
# Usage: sbatch tools/dex_teacher_v2_52k_steps.sbatch <STEPS>
#
# Deliberately does NOT run the converter: teacher_v2_stage2/ already holds the
# frozen 52000 conversion (job 486) — the generic dex_teacher_v2.sbatch would
# pick up the UNEVALUATED 54k leftover via `ls | tail -1`. Do not re-convert.
# Same seed (42) and procedure as jobs 468/486 -> meshes are paired.
set -euo pipefail
STEPS="${1:?usage: sbatch dex_teacher_v2_52k_steps.sbatch <STEPS>}"
INFER=/projects/gcaddeo/inference/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${INFER}"
echo "=== dex inference, frozen teacher 52k, steps=${STEPS} ==="
PYTHONPATH="${INFER}" \
DEX_STEPS="${STEPS}" \
DEX_OUT_ROOT="${INFER}/meshes_results_marching_cubes_teacher_v2_52k_s${STEPS}" \
python -u teacher_v2_port/inference_dex_teacher_v2.py
