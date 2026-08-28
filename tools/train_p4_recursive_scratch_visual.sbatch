#!/bin/bash
#SBATCH --job-name=p4s_visual
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# P4 FROM-SCRATCH methods control (user decision 2026-08-28): the recursive
# student trained from random init with EXACTLY student A's loss recipe
# (distill+physics, ni warmup 10k, NO visual) — the only delta vs A is the
# prior channel (gt_corrupt priors for the WHOLE run; never precomputed
# A-recons, which would leak A's knowledge into the from-scratch arm).
# 3 segments ~= 50k steps; eval its 48k ckpt vs A@48k (existing §7.8 evals)
# at matched steps: streaming P4 vs A+ring-buffer-median at equal K.
#
# Usage: sbatch tools/train_p4_recursive_scratch_visual.sbatch [<segment> <max>]
#        touch outputs/p4_recursive_scratch_visual/STOP_CHAIN to stop.
set -uo pipefail

SEGMENT="${1:-1}"
MAX_SEGMENTS="${2:-3}"

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
SBATCH=/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2_p4_recursive_scratch_visual.json"
OUTPUT_DIR="${REPO}/outputs/p4_recursive_scratch_visual"

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

mkdir -p "${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/STOP_CHAIN" ]; then
    echo "STOP_CHAIN sentinel present in ${OUTPUT_DIR}; exiting without training."
    exit 0
fi

if [ "${SEGMENT}" -lt "${MAX_SEGMENTS}" ]; then
    NEXT=$((SEGMENT + 1))
    if "${SBATCH}" --dependency=afterany:"${SLURM_JOB_ID}" \
           --job-name=p4s_visual \
           "${REPO}/tools/train_p4_recursive_scratch_visual.sbatch" "${NEXT}" "${MAX_SEGMENTS}"; then
        echo "Queued segment ${NEXT}/${MAX_SEGMENTS} after job ${SLURM_JOB_ID}"
    else
        echo "WARNING: could not queue segment ${NEXT}/${MAX_SEGMENTS}; the chain stops here." >&2
    fi
else
    echo "Segment ${SEGMENT}/${MAX_SEGMENTS} is the last one; not queuing a successor."
fi

nvidia-smi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ] || { echo "ERROR: DINOv2 not cached" >&2; exit 1; }

trap 'echo "SIGUSR1: 5 minutes to the wall clock"; kill -TERM ${TRAIN_PID} 2>/dev/null' USR1
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --load_dir "${OUTPUT_DIR}" \
    --ckpt latest \
    --num_gpus 2 \
    --auto_retry 3 \
    --data_dir "${DATA_TRAIN}" \
    --data_dir_test "${DATA_TEST}" &
TRAIN_PID=$!
wait ${TRAIN_PID}
RC=$?
echo "=== python exited with ${RC} ==="
exit ${RC}
