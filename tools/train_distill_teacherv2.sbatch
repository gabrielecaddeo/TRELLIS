#!/bin/bash
#SBATCH --job-name=distill_t2
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# Full distillation of the FROZEN teacher_v2 (EMA 52000) into the 8-block student
# (ICRA_PLAN.md item 3), as a self-chaining job — same pattern as
# tools/train_teacher_v2.sbatch. Smoke-tested 2026-08-20 (job 488, see
# outputs/distill_teacherv2_smoke/): teacher strict-loads from the FROZEN ckpt,
# ni_floor/contact_floor present, distill_mse 1.05 -> 0.22 in 400 steps, no NaN.
#
# DO NOT LAUNCH without user approval (ICRA_PLAN.md hard rule).
#
# Usage:
#   sbatch tools/train_distill_teacherv2.sbatch            # segment 1 of 3
#   sbatch tools/train_distill_teacherv2.sbatch 2 3        # <segment> <max_segments>
#
# To stop the chain early:
#   touch outputs/distill_teacherv2/STOP_CHAIN

set -uo pipefail

SEGMENT="${1:-1}"
MAX_SEGMENTS="${2:-3}"          # 3 x 24h ~ 45k steps at ~5-6 s/step

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
# --export=NONE leaves a minimal PATH without the Slurm bin dir; absolute path.
SBATCH=/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch

CONFIG="${REPO}/configs/generation/ss_flow_img_dit_S_16l8_fp16_sdf_conditioned_distill_teacherv2.json"
OUTPUT_DIR="${REPO}/outputs/distill_teacherv2"

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

mkdir -p "${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/STOP_CHAIN" ]; then
    echo "STOP_CHAIN sentinel present in ${OUTPUT_DIR}; exiting without training."
    exit 0
fi

# ---------------------------------------------------------------------------
# Queue the next segment now, dependent on this one finishing for any reason,
# so a SIGKILL at the 24h wall still leaves a successor queued.
# ---------------------------------------------------------------------------
if [ "${SEGMENT}" -lt "${MAX_SEGMENTS}" ]; then
    NEXT=$((SEGMENT + 1))
    if "${SBATCH}" --dependency=afterany:"${SLURM_JOB_ID}" \
           --job-name=distill_t2 \
           "${REPO}/tools/train_distill_teacherv2.sbatch" "${NEXT}" "${MAX_SEGMENTS}"; then
        echo "Queued segment ${NEXT}/${MAX_SEGMENTS} after job ${SLURM_JOB_ID}"
    else
        echo "WARNING: could not queue segment ${NEXT}/${MAX_SEGMENTS}; the chain stops here." >&2
    fi
else
    echo "Segment ${SEGMENT}/${MAX_SEGMENTS} is the last one; not queuing a successor."
fi

nvidia-smi

# --export=NONE means the interactive environment is gone; source conda explicitly.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
if [ ! -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]; then
    echo "ERROR: DINOv2 not cached at ${TORCH_HOME}/hub/facebookresearch_dinov2_main" >&2
    exit 1
fi

echo "=== distill segment=${SEGMENT}/${MAX_SEGMENTS} job=${SLURM_JOB_ID} node=$(hostname) ==="
echo "config=${CONFIG}"
echo "output=${OUTPUT_DIR}"

# Forward the pre-timeout signal to python so it can flush, then let SLURM kill us.
trap 'echo "SIGUSR1: 5 minutes to the wall clock"; kill -TERM ${TRAIN_PID} 2>/dev/null' USR1

# --load_dir OUTPUT_DIR --ckpt latest: cold start on segment 1 (no misc_*.pt yet
# -> load_ckpt stays None, fresh student), lossless resume on later segments.
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
