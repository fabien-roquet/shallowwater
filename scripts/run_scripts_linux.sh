#!/usr/bin/env bash
set -euo pipefail

# Run shallowwater example scripts on Linux.
# Usage:
#   ./scripts/run_scripts_linux.sh -a
#   ./scripts/run_scripts_linux.sh scripts/02_gravity_waves.py
#   ./scripts/run_scripts_linux.sh -l

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ALL=false
LIST_ONLY=false
PYTHON_RUNNER=(uv run python)

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS] [SCRIPT]

Options:
  -a, --all       Run all Python example scripts in scripts/
  -l, --list      List available scripts and exit
  -p, --python    Use plain python instead of 'uv run python'
  -h, --help      Show this help message

Examples:
  $0 --all
  $0 scripts/02_gravity_waves.py
  $0 --python --all
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
            ;;
        -p|--python)
            PYTHON_RUNNER=(python)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -* )
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

list_scripts() {
    find scripts -maxdepth 1 -type f -name '*.py' \
        ! -name 'strip_notebooks.py' \
        | sort
}

mkdir -p animations

if [ "$LIST_ONLY" = true ]; then
    list_scripts
    exit 0
fi

if [ "$RUN_ALL" = true ]; then
    list_scripts | while IFS= read -r script; do
        [ -z "$script" ] && continue
        echo "==> Running $script"
        "${PYTHON_RUNNER[@]}" "$script"
    done
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "Please provide a script path or use --all." >&2
    usage >&2
    exit 2
fi

script="$1"
if [ ! -f "$script" ]; then
    echo "Script not found: $script" >&2
    exit 1
fi

echo "==> Running $script"
"${PYTHON_RUNNER[@]}" "$script"
