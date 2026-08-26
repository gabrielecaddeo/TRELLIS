#!/bin/bash
#SBATCH --job-name=p4_recursive
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# P4 recursive student (EVAL_GUIDANCE.md §7.19, user-approved 2026-08-26):
# warm-start from the best single-view student, add the zero-init prior-latent
# channel (use_prior), train on simulated streaming (dataset builds the prior =
# warped median of OTHER views; stage 1 = corrupted GT, stage 2 = precomputed
# frozen-student recons via the _stage2 config after the precompute pass).
#
# Usage:
#   sbatch tools/train_p4_recursive.sbatch                    # segment 1 of 2
#   sbatch tools/train_p4_recursive.sbatch <segment> <max> [<config_suffix>]
#   touch outputs/p4_recursive/STOP_CHAIN                     # stop the chain
#
# INIT_CKPT: update if the §7.20 triple comparison re-picks the student.
set -uo pipefail

SEGMENT="${1:-1}"
MAX_SEGMENTS="${2:-2}"
CONFIG_SUFFIX="${3:-}"          # "" = stage-1 gt_corrupt; "_stage2" after precompute

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
SBATCH=/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2_p4_recursive${CONFIG_SUFFIX}.json"
OUTPUT_DIR="${REPO}/outputs/p4_recursive"
INIT_CKPT="${REPO}/outputs/distill_teacherv2/ckpts/denoiser_ema0.9999_step0081000.pt"

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
           --job-name=p4_recursive \
           "${REPO}/tools/train_p4_recursive.sbatch" "${NEXT}" "${MAX_SEGMENTS}" "${CONFIG_SUFFIX}"; then
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

EXTRA=""
if ! compgen -G "${OUTPUT_DIR}/ckpts/misc_*.pt" > /dev/null; then
    EXTRA="--load_flow_weights --ckpt_flow_path ${INIT_CKPT}"
    echo "Cold start: warm-starting from ${INIT_CKPT} (prior branch stays zero-init)"
fi

trap 'echo "SIGUSR1: 5 minutes to the wall clock"; kill -TERM ${TRAIN_PID} 2>/dev/null' USR1
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --load_dir "${OUTPUT_DIR}" \
    --ckpt latest \
    ${EXTRA} \
    --num_gpus 2 \
    --auto_retry 3 \
    --data_dir "${DATA_TRAIN}" \
    --data_dir_test "${DATA_TEST}" &
TRAIN_PID=$!
wait ${TRAIN_PID}
RC=$?
echo "=== python exited with ${RC} ==="
exit ${RC}
