#!/usr/bin/env bash
# Build the reproducible conda env for eh_shallow on the PIK HPC and pre-cache
# the observational inputs (compute nodes are offline, so this must run on a
# login node, which has internet).
#     bash scripts/cluster_setup.sh
set -eo pipefail   # not -u: module/conda init reference unset vars

PROJ=/p/projects/climber3/rostami/EH
ENV_PREFIX="$PROJ/envs/eh"
cd "$PROJ"
echo "[setup] host=$(hostname) proj=$PROJ"

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
conda config --set solver libmamba 2>/dev/null || true

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup] env exists -> updating"; conda env update -p "$ENV_PREFIX" -f environment.yml --prune
else
    echo "[setup] creating env at $ENV_PREFIX"; conda env create -p "$ENV_PREFIX" -f environment.yml
fi
conda activate "$ENV_PREFIX"
echo "[setup] python: $(python --version)  ($(which python))"

# Seed the land-mask cache from the shipped asset so grid.build() needs no
# network / regionmask (identical mask on every machine).
python - <<'PY'
import os, numpy as np
os.makedirs("eh_shallow/_cache", exist_ok=True)
m = np.load("eh_shallow/released/land_mask_0p5deg.npz")["mask"].astype(bool)
np.save("eh_shallow/_cache/land_mask_720x360.npy", m)
print("[setup] seeded land mask", m.shape, "land cells", int(m.sum()))
PY

# Pre-fetch the observational series on the login node (compute nodes are offline).
echo "[setup] pre-caching HadCRUT5 / AR6 ERF / NOAA OHC ..."
python - <<'PY'
from eh_shallow import data
print("  gmst   :", data.load_hadcrut5().attrs.get("source"))
print("  forcing:", data.load_ar6_erf().attrs.get("source"))
print("  ohc    :", data.load_ohc().attrs.get("source"))
PY
echo "[setup] DONE"
