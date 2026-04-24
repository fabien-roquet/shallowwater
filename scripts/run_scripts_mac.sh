#!/bin/sh
set -eu

# Run shallowwater example scripts on macOS.
# This version is compatible with the old /bin/sh and Bash 3.2 shipped by macOS.
# Usage:
#   ./scripts/run_scripts_mac.sh -a
#   ./scripts/run_scripts_mac.sh scripts/02_gravity_waves.py
#   ./scripts/run_scripts_mac.sh -l

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

RUN_ALL=false
LIST_ONLY=false
USE_UV=true

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS] [SCRIPT]

Options:
  -a, --all       Run all Python example scripts in scripts/
  -l, --list      List available scripts and exit
  -p, --python    Use plain python3 instead of 'uv run python'
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
            USE_UV=false
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

run_python_script() {
    script=$1
    echo "==> Running $script"
    if [ "$USE_UV" = true ]; then
        uv run python "$script"
    else
        python3 "$script"
    fi
}

mkdir -p animations

if [ "$LIST_ONLY" = true ]; then
    list_scripts
    exit 0
fi

if [ "$RUN_ALL" = true ]; then
    list_scripts | while IFS= read -r script; do
        [ -z "$script" ] && continue
        run_python_script "$script"
    done
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "Please provide a script path or use --all." >&2
    usage >&2
    exit 2
fi

script=$1
if [ ! -f "$script" ]; then
    echo "Script not found: $script" >&2
    exit 1
fi

run_python_script "$script"
