#!/bin/bash
#SBATCH --job-name=teacher_v2
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#
# Teacher v2 retrain (TEACHER_RETRAIN.md §5), as a self-chaining job.
#
# The gpu-h200 partition has a HARD 24-hour MaxTime for this account (verified with
# `sbatch --test-only`; there is no long-run QOS on it -- `gpuh-long` is not in our
# association). To get the requested ~72 hours of training the job re-submits itself:
# each segment runs 24h, checkpoints every `i_save` steps, and the next segment resumes
# with `--load_dir <output_dir> --ckpt latest`.
#
# Usage:
#   sbatch tools/train_teacher_v2.sbatch stage1                 # start segment 1 of 3
#   sbatch tools/train_teacher_v2.sbatch stage2                 # ditto, for stage 2
#   sbatch tools/train_teacher_v2.sbatch stage1 2 3             # <stage> <segment> <max_segments>
#
# To stop the chain early:
#   touch outputs/teacher_v2_<stage>/STOP_CHAIN
# The next segment sees the sentinel and exits without training or re-queuing.

set -uo pipefail

STAGE="${1:-stage1}"
SEGMENT="${2:-1}"
MAX_SEGMENTS="${3:-3}"          # 3 x 24h = 72h, the wall-clock budget asked for

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
# --export=NONE leaves a minimal PATH that does not include the Slurm bin dir, so the
# self-resubmission below has to call sbatch by absolute path.
SBATCH=/opt/share/sw/amd/gcc-11.4.1/slurm-24.11.7/bin/sbatch

BASE_CKPT=/projects/gcaddeo/train_flow/TRELLIS/outputs/flow_final_norm_resume330/ckpts/denoiser_ema0.9999_step0350000.pt

DATA_TRAIN="${REPO}/datasets_split/Leap_Hand_train,${REPO}/datasets_split/Hands_train,${REPO}/datasets_split/Hands_Google_train"
DATA_TEST="${REPO}/datasets_split/Leap_Hand_test,${REPO}/datasets_split/Hands_test,${REPO}/datasets_split/Hands_Google_test"

case "${STAGE}" in
    stage1)
        CONFIG="${REPO}/configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage1.json"
        OUTPUT_DIR="${REPO}/outputs/teacher_v2_stage1_cond"
        # Stage 1 warm-starts from the pre-conditioning base checkpoint. `train.py` only
        # honours --load_flow_weights on a fresh output dir: once ckpts/misc_*.pt exists,
        # find_ckpt() resolves `--ckpt latest` and Trainer.load() overwrites the warm start.
        # So the base-checkpoint flags are only passed on the very first segment.
        EXTRA_FIRST="--load_flow_weights --initialize_layers --ckpt_flow_path ${BASE_CKPT}"
        ;;
    stage2)
        CONFIG="${REPO}/configs/generation/ss_flow_img_dit_L_16l8_fp16_sdf_conditioned_stage2.json"
        OUTPUT_DIR="${REPO}/outputs/teacher_v2_stage2_physics"
        # Stage 2 resumes from stage 1's final checkpoint. Handled below via LOAD_DIR.
        EXTRA_FIRST=""
        ;;
    *)
        echo "ERROR: unknown stage '${STAGE}' (expected stage1 or stage2)" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/STOP_CHAIN" ]; then
    echo "STOP_CHAIN sentinel present in ${OUTPUT_DIR}; exiting without training."
    exit 0
fi

# Has this run ever checkpointed? If yes we resume, regardless of segment number --
# this also makes a manual re-submit after a crash do the right thing.
HAS_CKPT=0
if compgen -G "${OUTPUT_DIR}/ckpts/misc_*.pt" > /dev/null; then
    HAS_CKPT=1
fi

if [ "${HAS_CKPT}" -eq 1 ]; then
    LOAD_ARGS="--load_dir ${OUTPUT_DIR} --ckpt latest"
    EXTRA=""
    echo "Resuming from the latest checkpoint in ${OUTPUT_DIR}"
elif [ "${STAGE}" = "stage2" ]; then
    LOAD_ARGS="--load_dir ${REPO}/outputs/teacher_v2_stage1_cond --ckpt latest"
    EXTRA=""
    NEEDS_WARMUP_FIX=1
    echo "Stage 2 cold start: resuming from stage 1"
else
    LOAD_ARGS="--load_dir ${OUTPUT_DIR} --ckpt latest"   # no misc_*.pt yet -> load_ckpt stays None
    EXTRA="${EXTRA_FIRST}"
    echo "Stage 1 cold start: warm-starting from ${BASE_CKPT}"
fi

# ---------------------------------------------------------------------------
# Queue the next segment now, dependent on this one finishing for any reason.
# Doing it up-front (rather than after training) means a SIGKILL at the 24h wall
# still leaves a successor queued.
# ---------------------------------------------------------------------------
if [ "${SEGMENT}" -lt "${MAX_SEGMENTS}" ]; then
    NEXT=$((SEGMENT + 1))
    if "${SBATCH}" --dependency=afterany:"${SLURM_JOB_ID}" \
           --job-name="teacher_v2_${STAGE}" \
           "${REPO}/tools/train_teacher_v2.sbatch" "${STAGE}" "${NEXT}" "${MAX_SEGMENTS}"; then
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

# The lambda_ni warmup is compared against the *resumed* step counter, so on a stage-2
# cold start it has to be pushed past stage 1's final step or lambda jumps to max at once.
if [ "${NEEDS_WARMUP_FIX:-0}" -eq 1 ]; then
    python tools/set_stage2_warmup.py \
        --stage1_dir "${REPO}/outputs/teacher_v2_stage1_cond" \
        --config "${CONFIG}" \
        --ramp 2000 || exit 1
fi

echo "=== stage=${STAGE} segment=${SEGMENT}/${MAX_SEGMENTS} job=${SLURM_JOB_ID} node=$(hostname) ==="
echo "config=${CONFIG}"
echo "output=${OUTPUT_DIR}"

# Forward the pre-timeout signal to python so it can flush, then let SLURM kill us.
trap 'echo "SIGUSR1: 5 minutes to the wall clock"; kill -TERM ${TRAIN_PID} 2>/dev/null' USR1

# -u: stdout to a file is block-buffered, which held "Step N completed" prints back for
# tens of minutes and made a healthy run look stalled. Progress lines are worth the flushes.
python -u train.py \
    --config "${CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    ${LOAD_ARGS} \
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
