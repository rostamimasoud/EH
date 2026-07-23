#!/usr/bin/env bash
#SBATCH --job-name=eh_shallow
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/p/projects/climber3/rostami/EH/outputs/slurm-%j.out
#SBATCH --error=/p/projects/climber3/rostami/EH/outputs/slurm-%j.err
#
# Full-resolution EH pipeline run. Submit from a login node:
#     sbatch scripts/slurm_run.sh
# Override the SMC ensemble size:  sbatch scripts/slurm_run.sh --particles 800
set -eo pipefail   # not -u: module/conda init reference unset vars

PROJ=/p/projects/climber3/rostami/EH
ENV_PREFIX="$PROJ/envs/eh"
mkdir -p "$PROJ/outputs"
cd "$PROJ"

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
             /usr/share/lmod/lmod/init/bash; do
        [ -r "$f" ] && source "$f" && break
    done
fi
module purge 2>/dev/null || true
for mp in compiler gpu libraries parallel tools; do
    module use "/p/system/modulefiles_rhel9/$mp" 2>/dev/null || true
done
module load anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Keep numeric libs from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export EH_OUTPUT_DIR="$PROJ/outputs"
# No manuscript tree on the cluster: leave EH_PAPER_FIG_DIR unset so the figure
# module writes to outputs/figures only (mirroring is skipped when the paper dir
# is absent or identical to FIG_DIR). Figures are retrieved locally via rsync.

echo "[job] $(date) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} python=$(which python)"
srun python scripts/run_pipeline.py "$@"
echo "[job] $(date) done"
