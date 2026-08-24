#!/bin/bash
#SBATCH --job-name=verify_fixes
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
# `python tools/x.py` puts tools/ on sys.path, not the repo root.
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "############ WITH the fix ############"
python tools/verify_fixes.py "$@"
RC=$?

echo
echo "############ WITHOUT the hand positional embedding (expect check 4 to FAIL) ############"
python tools/verify_fixes.py --no_hand_pe "$@"

exit ${RC}
