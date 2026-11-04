#!/bin/bash
# One-time setup for the SrTiO3 benchmark on an Apple-silicon Mac.
#
#   cd ~/Documents/srtio3-mlip-benchmark
#   bash setup_mac.sh                 # all five model environments
#   bash setup_mac.sh mace orb        # only these
#
# What it does (nothing is installed system-wide, and Anaconda is left alone):
#   1. checks you are on native Apple silicon and have >= 20 GB free
#   2. installs uv (a fast Python package manager) into ~/.local/bin
#   3. installs Python 3.12 through uv
#   4. makes one environment per model family in ~/.sto-envs/ (kept outside Documents so
#      iCloud never tries to sync thousands of package files)
#   5. runs the smoke test in each, which downloads every model's weights, and the unit tests
#   6. records exact package versions in env/locked/ for the paper trail
# Re-running is safe: finished steps are skipped. To undo everything:
#   rm -rf ~/.sto-envs ~/.cache/uv ~/.cache/mace ~/.cache/orb ~/.local/mattersim
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
ENVS="${STO_ENVS:-$HOME/.sto-envs}"
PY="3.12"
MIN_FREE_GB=20
if [ "$#" -gt 0 ]; then FAMILIES="$*"; else FAMILIES="mace orb mattersim sevenn chgnet"; fi
LOG="$REPO/logs"
mkdir -p "$LOG" "$REPO/env/locked"
chmod +x "$REPO/run.sh" "$REPO/setup_mac.sh" 2>/dev/null || true
export UV_PYTHON_PREFERENCE=only-managed  # never pick up Anaconda's Python by accident

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31mSTOP: %s\033[0m\n' "$*"; exit 1; }

# 1a. Native Apple silicon (an Intel-mode Terminal would pull Intel wheels, which torch no longer ships)
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ] \
   && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  say "This Terminal is running in Intel (Rosetta) mode; restarting natively"
  exec arch -arm64 /bin/bash "$0" "$@"
fi

# 1b. Disk space
free_gb=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1048576)}')
say "Free disk space: ${free_gb} GB"
if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ] && [ -z "${STO_SKIP_DISK_CHECK:-}" ]; then
  die "only ${free_gb} GB free. This needs about ${MIN_FREE_GB} GB: ~6 GB of environments, ~2 GB of
model weights, results, and room for macOS to swap (an 8 GB Mac swaps a lot). Free some space and run again."
fi

# 1c. Apple's command line tools (a compiler, for any package without a ready-made wheel)
if [ "$(uname -s)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install >/dev/null 2>&1 || true
  die "Apple's command line tools are being installed (a window should have opened).
Run this script again when that finishes."
fi

# 2. uv
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh \
    || die "could not download uv. Check the internet connection and run again."
fi
echo "uv: $(uv --version)"

# 3. Python
say "Python $PY (managed by uv)"
uv python install "$PY" || die "could not install Python $PY"

# 4-6. One environment per family
mkdir -p "$ENVS"
ok=""
bad=""
for fam in $FAMILIES; do
  req="$REPO/env/$fam.txt"
  if [ ! -f "$req" ]; then
    echo "No requirements file for '$fam' (expected $req). Known: mace orb mattersim sevenn chgnet"
    bad="$bad $fam"
    continue
  fi
  py="$ENVS/$fam/bin/python"
  say "[$fam] environment: $ENVS/$fam"
  if [ ! -x "$py" ]; then
    if ! uv venv --python "$PY" "$ENVS/$fam" >>"$LOG/setup_$fam.log" 2>&1; then
      echo "  could not create it; see logs/setup_$fam.log"
      bad="$bad $fam"
      continue
    fi
  fi
  say "[$fam] installing packages (first time: a few minutes)"
  if ! uv pip install --python "$py" -r "$req" 2>&1 | tee -a "$LOG/setup_$fam.log"; then
    echo "  install failed; see logs/setup_$fam.log"
    bad="$bad $fam"
    continue
  fi
  uv pip check --python "$py" >>"$LOG/setup_$fam.log" 2>&1 \
    || echo "  note: 'uv pip check' reported a version conflict (details in logs/setup_$fam.log)"
  uv pip freeze --python "$py" >"$REPO/env/locked/$fam.txt" 2>/dev/null || true

  say "[$fam] smoke test (downloads the model weights the first time)"
  if (cd "$REPO" && "$py" scripts/02_smoke_test.py --family "$fam" 2>&1 | tee "$LOG/smoke_$fam.log"); then
    ok="$ok $fam"
  else
    bad="$bad $fam"
  fi
done

if [ -x "$ENVS/mace/bin/python" ]; then
  say "Unit tests"
  (cd "$REPO" && "$ENVS/mace/bin/python" -m pytest -q tests 2>&1 | tee "$LOG/tests.log") || bad="$bad tests"
fi

say "Summary"
echo "  working: ${ok:- none}"
echo "  failed:  ${bad:- none}"
echo "  environments in $ENVS; logs in $REPO/logs"
df -h "$HOME" | awk 'NR==2 {print "  free space now: " $4}'
if [ -n "$bad" ]; then
  echo
  echo "Send me the files in logs/ for the failed parts and I will fix them."
  exit 1
fi
echo
echo "All set. Next: python3 scripts/00_audit.py /path/to/vasp/archive"
