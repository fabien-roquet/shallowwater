#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run Jupyter notebooks in-place (execute & overwrite).
# Requires: jupyter nbconvert (comes with Jupyter)
#
# Usage:
#   ./run_notebooks.sh -n 01_example.ipynb
#   ./run_notebooks.sh -a
#   ./run_notebooks.sh -a -t 7200 -k python3
#   ./run_notebooks.sh -a -x "*WIP*.ipynb" -x "*draft*.ipynb"
#   ./run_notebooks.sh -a --continue-on-error
#
# Options:
#   -n, --notebook FILE     Run only this notebook.
#   -a, --all               Run all *.ipynb in the current directory.
#   -t, --timeout SECONDS   Cell execution timeout (default: 3600).
#   -k, --kernel NAME       Kernel name to use (e.g. python3). Optional.
#   -x, --exclude PATTERN   Glob(s) to skip (can be used multiple times).
#   -c, --continue-on-error Continue even if a cell errors (nbconvert --allow-errors).
#   -h, --help              Show this message.

usage() {
  sed -n '1,100p' "$0" | sed -n '5,999p' | sed '/^usage() {/,/^}/d' | sed '/^# Run/d'
  exit "${1:-0}"
}

NOTEBOOK=""
RUN_ALL=false
TIMEOUT=3600
KERNEL=""
CONTINUE=false
EXCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--notebook) NOTEBOOK="${2:-}"; shift 2 ;;
    -a|--all) RUN_ALL=true; shift ;;
    -t|--timeout) TIMEOUT="${2:-}"; shift 2 ;;
    -k|--kernel) KERNEL="${2:-}"; shift 2 ;;
    -x|--exclude) EXCLUDES+=("${2:-}"); shift 2 ;;
    -c|--continue-on-error) CONTINUE=true; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1"; usage 1 ;;
  esac
done

if [[ -z "$NOTEBOOK" && "$RUN_ALL" = false ]]; then
  echo "Pick one: --notebook FILE or --all"; usage 1
fi

# Build nbconvert options
NBC_OPTS=( --to notebook --inplace --execute --ExecutePreprocessor.timeout="$TIMEOUT" )
[[ -n "$KERNEL" ]] && NBC_OPTS+=( --ExecutePreprocessor.kernel_name="$KERNEL" )
[[ "$CONTINUE" = true ]] && NBC_OPTS+=( --allow-errors )

# helper: match against exclusions
is_excluded() {
  local f="$1"
  for pat in "${EXCLUDES[@]:-}"; do
    [[ "$f" == $pat ]] && return 0
  done
  # also skip ipynb_checkpoints
  [[ "$f" == *".ipynb_checkpoints"* ]] && return 0
  return 1
}

run_one() {
  local f="$1"
  echo ">>> Running: $f"
  jupyter nbconvert "${NBC_OPTS[@]}" "$f"
  echo "✔  Done:     $f"
}

if [[ -n "$NOTEBOOK" ]]; then
  if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Notebook not found: $NOTEBOOK" >&2; exit 1
  fi
  if is_excluded "$NOTEBOOK"; then
    echo "Notebook excluded by pattern(s): $NOTEBOOK" >&2; exit 1
  fi
  run_one "$NOTEBOOK"
  exit 0
fi

# RUN_ALL = true
# Gather notebooks in current directory only (no subfolders)
mapfile -t nbs < <(printf "%s\n" ./*.ipynb | sort || true)

if [[ ${#nbs[@]} -eq 0 ]]; then
  echo "No notebooks (*.ipynb) found in $(pwd)"; exit 0
fi

any=false
for nb in "${nbs[@]}"; do
  [[ "$nb" == "./*.ipynb" ]] && continue  # in case glob didn't match
  if is_excluded "$nb"; then
    echo "Skipping (excluded): $nb"
    continue
  fi
  any=true
  run_one "$nb"
done

if [[ "$any" = false ]]; then
  echo "No notebooks to run (all excluded)"; exit 0
fi
