#!/bin/bash
#SBATCH --job-name=distill_s8m4
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# Architecture ablation (EVAL_GUIDANCE.md 7.8): 8-block / mlp_ratio-4 student,
# COPY-INITIALIZED from every 3rd frozen-teacher block (vs the deployed 8x mlp2
# from-scratch student). Same distill recipe/config otherwise. Self-chaining like
# tools/train_distill_teacherv2.sbatch.
#
# Usage: sbatch tools/train_distill_teacherv2_s8mlp4.sbatch            # seg 1 of 2
#        sbatch tools/train_distill_teacherv2_s8mlp4.sbatch 2 2
# Stop:  touch outputs/distill_s8mlp4_copyinit/STOP_CHAIN

set -uo pipefail
SEGMENT="${1:-1}"
MAX_SEGMENTS="${2:-2}"

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
SBATCH=/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S8mlp4_16l8_fp16_sdf_conditioned_distill_teacherv2.json"
OUTPUT_DIR="${REPO}/outputs/distill_s8mlp4_copyinit"
INIT_CKPT="${OUTPUT_DIR}/student_init_from_teacher.pt"

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

mkdir -p "${OUTPUT_DIR}"
if [ -f "${OUTPUT_DIR}/STOP_CHAIN" ]; then
    echo "STOP_CHAIN present; exiting."; exit 0
fi

if [ "${SEGMENT}" -lt "${MAX_SEGMENTS}" ]; then
    NEXT=$((SEGMENT + 1))
    "${SBATCH}" --dependency=afterany:"${SLURM_JOB_ID}" --job-name=distill_s8m4 \
        "${REPO}/tools/train_distill_teacherv2_s8mlp4.sbatch" "${NEXT}" "${MAX_SEGMENTS}" \
        && echo "Queued segment ${NEXT}/${MAX_SEGMENTS}" \
        || echo "WARNING: could not queue segment ${NEXT}" >&2
fi

nvidia-smi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ] || { echo "ERROR: DINOv2 not cached" >&2; exit 1; }

# Warm start only applies to a fresh run; once ckpts exist, resume wins (train.py
# only honours --load_flow_weights when find_ckpt resolves nothing).
EXTRA=""
if ! compgen -G "${OUTPUT_DIR}/ckpts/misc_*.pt" > /dev/null; then
    if [ ! -f "${INIT_CKPT}" ]; then
        python tools/make_student_init_from_teacher.py --out "${INIT_CKPT}" || exit 1
    fi
    EXTRA="--load_flow_weights --ckpt_flow_path ${INIT_CKPT}"
    echo "Cold start: copy-init from ${INIT_CKPT}"
else
    echo "Resuming from latest checkpoint in ${OUTPUT_DIR}"
fi

echo "=== s8mlp4 ablation segment=${SEGMENT}/${MAX_SEGMENTS} job=${SLURM_JOB_ID} node=$(hostname) ==="
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
