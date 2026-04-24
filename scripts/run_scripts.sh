#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Run converted tutorial scripts and write animations to ../animations/.
#
# Usage:
#   ./scripts/run_scripts.sh -s scripts/01_wind_gyre.py
#   ./scripts/run_scripts.sh -a
#   ./scripts/run_scripts.sh -a -x "*Rossby*" --continue-on-error
#
# Options:
#   -s, --script FILE       Run only this script.
#   -a, --all               Run all converted tutorial scripts in scripts/.
#   -x, --exclude PATTERN   Glob(s) to skip; repeatable.
#   -c, --continue-on-error Continue even if one script fails.
#   -h, --help              Show this message.

usage() {
  sed -n '1,80p' "$0" | sed -n '5,999p' | sed '/^usage() {/,/^}/d'
  exit "${1:-0}"
}

SCRIPT=""
RUN_ALL=false
CONTINUE=false
EXCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--script) SCRIPT="${2:-}"; shift 2 ;;
    -a|--all) RUN_ALL=true; shift ;;
    -x|--exclude) EXCLUDES+=("${2:-}"); shift 2 ;;
    -c|--continue-on-error) CONTINUE=true; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$SCRIPT" && "$RUN_ALL" = false ]]; then
  echo "Pick one: --script FILE or --all" >&2
  usage 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
mkdir -p "$REPO_ROOT/animations"

is_excluded() {
  local f="$1"
  for pat in "${EXCLUDES[@]:-}"; do
    [[ "$(basename "$f")" == $pat || "$f" == $pat ]] && return 0
  done
  return 1
}

run_one() {
  local f="$1"
  echo ">>> Running: $f"
  if [[ "$CONTINUE" = true ]]; then
    python "$f" || echo "!!! Failed: $f" >&2
  else
    python "$f"
  fi
  echo "Done: $f"
}

if [[ -n "$SCRIPT" ]]; then
  [[ -f "$SCRIPT" ]] || { echo "Script not found: $SCRIPT" >&2; exit 1; }
  is_excluded "$SCRIPT" && { echo "Script excluded: $SCRIPT" >&2; exit 1; }
  run_one "$SCRIPT"
  exit 0
fi

mapfile -t files < <(find "$THIS_DIR" -maxdepth 1 -type f -name '[0-9][0-9]_*.py' | sort)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No converted tutorial scripts found in $THIS_DIR"
  exit 0
fi

for f in "${files[@]}"; do
  if is_excluded "$f"; then
    echo "Skipping excluded script: $f"
    continue
  fi
  run_one "$f"
done
