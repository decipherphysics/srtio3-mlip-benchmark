#!/bin/bash
# Run one script once in every model environment, keeping the Mac awake while it runs.
#
#   ./run.sh scripts/02_smoke_test.py
#   ./run.sh scripts/04_strain_relax.py --quick
#   STO_FAMILIES="mace orb" ./run.sh scripts/04_strain_relax.py
#
# Each environment's output is also saved to logs/. Scripts 00, 01 and 06 do not need this:
# run them directly with  ~/.sto-envs/mace/bin/python scripts/<script>.py
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
ENVS="${STO_ENVS:-$HOME/.sto-envs}"
FAMILIES="${STO_FAMILIES:-mace orb mattersim sevenn chgnet}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run.sh scripts/<script>.py [options]"
  exit 1
fi
if command -v caffeinate >/dev/null 2>&1 && [ -z "${STO_AWAKE:-}" ]; then
  export STO_AWAKE=1
  exec caffeinate -i /bin/bash "$0" "$@"  # stops the Mac sleeping until this finishes
fi

script="$1"
shift
mkdir -p "$REPO/logs"
stamp=$(date +%Y%m%d-%H%M%S)
for fam in $FAMILIES; do
  py="$ENVS/$fam/bin/python"
  if [ ! -x "$py" ]; then
    echo "--- $fam: no environment at $py (run setup_mac.sh), skipped"
    continue
  fi
  echo "=== $fam ==="
  log="$REPO/logs/$(basename "$script" .py)_${fam}_${stamp}.log"
  (cd "$REPO" && "$py" "$script" --family "$fam" ${@+"$@"} 2>&1 | tee "$log") \
    || echo "!!! $fam stopped with an error (see $log); carrying on with the next environment"
done
