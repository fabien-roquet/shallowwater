#!/usr/bin/env python3
"""Strip outputs and volatile execution metadata from Jupyter notebooks.

Usage:
    python scripts/strip_notebooks.py
    python scripts/strip_notebooks.py notebooks/*.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VOLATILE_CELL_METADATA = {
    "execution",
    "ExecuteTime",
    "collapsed",
    "scrolled",
}
VOLATILE_NB_METADATA = {
    "widgets",
}


def strip_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
        meta = cell.get("metadata", {})
        for key in list(meta):
            if key in VOLATILE_CELL_METADATA:
                meta.pop(key, None)
                changed = True

    meta = data.get("metadata", {})
    for key in list(meta):
        if key in VOLATILE_NB_METADATA:
            meta.pop(key, None)
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(Path(".").rglob("*.ipynb"))
        paths = [p for p in paths if ".ipynb_checkpoints" not in p.parts]

    changed = [p for p in paths if p.exists() and strip_notebook(p)]
    if changed:
        print("Stripped notebook outputs:")
        for p in changed:
            print(f"  {p}")
    else:
        print("No notebook outputs to strip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
