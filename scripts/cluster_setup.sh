#!/usr/bin/env bash
# Build the reproducible conda env for the EH pipeline on the PIK HPC.
# Run on a login node:  bash scripts/cluster_setup.sh
set -eo pipefail   # not -u: module/conda init reference unset vars

PROJ=/p/projects/climber3/rostami/EH
ENV_PREFIX="$PROJ/envs/eh"

cd "$PROJ"
echo "[setup] host=$(hostname) proj=$PROJ"

# `module` is a shell function defined by the environment-modules/lmod init
# script; a non-login shell (e.g. nohup bash) does not source it. Load it.
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
             /usr/share/lmod/lmod/init/bash; do
        [ -r "$f" ] && source "$f" && break
    done
fi
module purge 2>/dev/null || true
# Non-login shells miss the site MODULEPATH the login profile adds; add it.
for mp in compiler gpu libraries parallel tools; do
    module use "/p/system/modulefiles_rhel9/$mp" 2>/dev/null || true
done
module load anaconda/2025
# Make `conda activate` work in a non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[setup] conda: $(conda --version)"
conda config --set solver libmamba 2>/dev/null || true

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup] env already exists at $ENV_PREFIX — updating"
    conda env update -p "$ENV_PREFIX" -f environment.yml --prune
else
    echo "[setup] creating env at $ENV_PREFIX"
    conda env create -p "$ENV_PREFIX" -f environment.yml
fi

conda activate "$ENV_PREFIX"
echo "[setup] python: $(python --version)  ($(which python))"
python - <<'PY'
import numpy, scipy, pandas, sklearn, matplotlib, cartopy, shapely, requests, PyCO2SYS
print("[setup] imports OK:",
      "numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| sklearn", sklearn.__version__, "| cartopy", cartopy.__version__)
PY
echo "[setup] DONE"
