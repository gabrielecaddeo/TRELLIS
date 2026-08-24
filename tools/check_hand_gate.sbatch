#!/bin/bash
#SBATCH --job-name=hand_gate
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
#
# Stage-1 -> stage-2 gate. Runs probe C on the newest stage-1 EMA checkpoint and prints a
# READY / EMERGING / NOT READY verdict. ~6 minutes. Safe to run while training continues --
# it only reads checkpoints and writes under outputs/diagnostics/.
#
#   sbatch tools/check_hand_gate.sbatch                                  # newest checkpoint
#   sbatch tools/check_hand_gate.sbatch --ckpt denoiser_ema0.9999_step0006000.pt
#
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
RUN_DIR="${RUN_DIR:-outputs/teacher_v2_stage1_cond}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

WEIGHTS="${WEIGHTS:-raw}"
CKPT="$(python -u tools/check_hand_gate.py --run_dir "${RUN_DIR}" --weights "${WEIGHTS}" --latest_ckpt)" || exit 1
echo "gating on ${RUN_DIR}/ckpts/${CKPT}"

# Held-out split, so the gate is not measured on training data.
python -u tools/diagnose_physics_losses.py \
    --teacher_dir "${RUN_DIR}" \
    --ckpt        "${CKPT}" \
    --probes      C \
    --output      "outputs/diagnostics/handgate_$(basename ${RUN_DIR})_${CKPT%.pt}.json" \
    --num_batches 40 \
    --data_dir    "${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test" \
    "$@" || exit 1

echo; echo "================ GATE VERDICT ================"
python -u tools/check_hand_gate.py --run_dir "${RUN_DIR}" --ckpt "${CKPT}" --weights "${WEIGHTS}"
