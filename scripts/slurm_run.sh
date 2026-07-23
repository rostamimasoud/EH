#!/usr/bin/env bash
#SBATCH --job-name=eh_shallow
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/p/projects/climber3/rostami/EH/outputs/slurm-%j.out
#SBATCH --error=/p/projects/climber3/rostami/EH/outputs/slurm-%j.err
#
# Full eh_shallow run. Run scripts/cluster_setup.sh once first (builds the env,
# seeds the land mask, and pre-caches the observational series on a login node).
#     sbatch scripts/slurm_run.sh                       # headline SSP2-4.5
#     sbatch scripts/slurm_run.sh --n-particles 800     # override
set -eo pipefail

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Ensure the land-mask cache exists (idempotent; setup normally does this).
python - <<'PY'
import os, numpy as np
c = "eh_shallow/_cache/land_mask_720x360.npy"
if not os.path.exists(c):
    os.makedirs("eh_shallow/_cache", exist_ok=True)
    m = np.load("eh_shallow/released/land_mask_0p5deg.npz")["mask"].astype(bool)
    np.save(c, m); print("[job] seeded land mask")
PY

echo "[job] $(date) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} python=$(which python)"
srun python -m eh_shallow.run --n-particles 400 --baseline auto \
     --outdir "$PROJ/outputs" "$@"
echo "[job] $(date) done"
