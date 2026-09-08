#!/bin/bash
#SBATCH --job-name=dexfull_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out
# Generic CPU runner for dex-full tooling (the login node caps RSS/virtual memory and
# kills the calibration fits). Usage: sbatch tools/dexfull_cpu.sbatch <command...>
set -uo pipefail
REPO=/projects/gcaddeo/train_flow_conditioned/TRELLIS
CONDA_ROOT=/opt/share/sw/amd/gcc-11.4.1/miniforge3-24.11.3-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate trellis_hopper
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
echo "=== $(date) :: $*"
"$@"
echo "=== done $(date) rc=$?"
