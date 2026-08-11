#!/usr/bin/env python3
"""Validate generated course material and optionally execute/compile it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

import nbformat
import numpy as np


COURSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COURSE_ROOT.parent
NOTEBOOK_DIR = COURSE_ROOT / "notebooks"
DATA_DIR = COURSE_ROOT / "data"
LECTURE_DIR = COURSE_ROOT / "lecture"
EXPECTED_VERSION = "0.1.4"

NOTEBOOKS = (
    "part_a_waves_student.ipynb",
    "part_a_waves_solutions.ipynb",
    "part_b_bathymetry_student.ipynb",
    "part_b_bathymetry_solutions.ipynb",
    "part_c_project.ipynb",
)


def fail(message):
    raise RuntimeError(message)


def read_notebook(name):
    path = NOTEBOOK_DIR / name
    if not path.exists():
        fail(f"Missing notebook: {path}")
    return nbformat.read(path, as_version=4)


def validate_package_version():
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if project["version"] != EXPECTED_VERSION:
        fail(
            f"pyproject version is {project['version']!r}, expected {EXPECTED_VERSION!r}."
        )


def validate_notebooks():
    loaded = {name: read_notebook(name) for name in NOTEBOOKS}
    absolute_path_pattern = re.compile(r"/Users/|[A-Za-z]:\\Users\\")

    for name, notebook in loaded.items():
        ids = [cell.get("id") for cell in notebook.cells]
        if len(ids) != len(set(ids)) or any(not cell_id for cell_id in ids):
            fail(f"Notebook has missing or duplicate cell ids: {name}")
        if notebook.metadata.get("course", {}).get("package_version") != EXPECTED_VERSION:
            fail(f"Notebook package version metadata is wrong: {name}")
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell-{index}", "exec")
                if cell.get("outputs"):
                    fail(f"Generated source notebook contains outputs: {name}, cell {index}")
                if cell.get("execution_count") is not None:
                    fail(f"Generated source notebook has execution count: {name}, cell {index}")
            if absolute_path_pattern.search(cell.source):
                fail(f"Absolute user path found in {name}, cell {index}")

    for part in ("part_a_waves", "part_b_bathymetry"):
        student = loaded[f"{part}_student.ipynb"]
        solution = loaded[f"{part}_solutions.ipynb"]
        if [c.id for c in student.cells] != [c.id for c in solution.cells]:
            fail(f"Student/solution structure differs for {part}")
        for student_cell, solution_cell in zip(student.cells, solution.cells):
            if student_cell.cell_type != solution_cell.cell_type:
                fail(f"Student/solution cell type differs for {part}")
            if student_cell.cell_type == "code" and student_cell.source != solution_cell.source:
                fail(f"Student/solution code differs for {part}, cell {student_cell.id}")
        student_text = "\n".join(cell.source for cell in student.cells)
        solution_text = "\n".join(cell.source for cell in solution.cells)
        if "**Solution.**" in student_text or '"solution"' in student_text:
            fail(f"Solution content leaked into {part} student notebook")
        if "**Solution.**" not in solution_text:
            fail(f"No solution responses found in {part} solution notebook")

    part_c_text = "\n".join(cell.source for cell in loaded["part_c_project.ipynb"].cells)
    required = (
        "## Group and research question",
        "## Prediction",
        "## Baseline configuration",
        "## Controlled variations",
        "## Diagnostics",
        "## Results",
        "## Comparison with theory",
        "## Limitations",
        "## Conclusion",
        "## Contributions and submission check",
    )
    missing = [heading for heading in required if heading not in part_c_text]
    if missing:
        fail(f"Part C is missing required headings: {missing}")


def validate_data():
    with np.load(DATA_DIR / "example_bathymetry.npz", allow_pickle=False) as archive:
        H = np.asarray(archive["H"])
        if H.shape != (24, 160) or not np.isfinite(H).all() or np.min(H) <= 0:
            fail("Example bathymetry is invalid")
    with np.load(DATA_DIR / "example_wind_forcing.npz", allow_pickle=False) as archive:
        tau_x = np.asarray(archive["tau_x"])
        tau_y = np.asarray(archive["tau_y"])
        if tau_x.shape != (24, 160) or tau_y.shape != (24, 160):
            fail("Example wind forcing has the wrong shape")
        if not np.isfinite(tau_x).all() or not np.isfinite(tau_y).all():
            fail("Example wind forcing contains non-finite values")


def validate_lecture_files():
    tex = LECTURE_DIR / "shallow_water_waves_lecture.tex"
    pdf = LECTURE_DIR / "shallow_water_waves_lecture.pdf"
    if not tex.exists() or not pdf.exists() or pdf.stat().st_size < 10_000:
        fail("Lecture source or compiled PDF is missing")
    frame_count = tex.read_text(encoding="utf-8").count("\\begin{frame}")
    if not 20 <= frame_count <= 28:
        fail(f"Unexpected main lecture frame count: {frame_count}")


def strip_notebook_outputs():
    """Remove accidental outputs from generated distribution notebooks."""
    for name in NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        notebook = nbformat.read(path, as_version=4)
        changed = False
        for cell in notebook.cells:
            if cell.cell_type == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    cell.outputs = []
                    cell.execution_count = None
                    changed = True
        if changed:
            nbformat.write(notebook, path)
            print(f"Stripped outputs: {path.name}")


def execute_notebooks():
    with tempfile.TemporaryDirectory(prefix="shallowwater-course-") as output_dir:
        env = os.environ.copy()
        env.setdefault("SHALLOWWATER_USE_NUMBA", "0")
        env.setdefault("MPLCONFIGDIR", str(Path(output_dir) / "matplotlib"))
        env.setdefault("JUPYTER_CONFIG_DIR", str(Path(output_dir) / "jupyter"))
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--ExecutePreprocessor.timeout=600",
            f"--output-dir={output_dir}",
            *NOTEBOOKS,
        ]
        subprocess.run(command, cwd=NOTEBOOK_DIR, env=env, check=True)
        for name in NOTEBOOKS:
            executed = nbformat.read(Path(output_dir) / name, as_version=4)
            for cell in executed.cells:
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        fail(f"Execution error in {name}: {output.get('evalue')}")


def compile_lecture():
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "shallow_water_waves_lecture.tex",
        ],
        cwd=LECTURE_DIR,
        check=True,
    )
    log = (LECTURE_DIR / "shallow_water_waves_lecture.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "Overfull \\hbox" in log or "Overfull \\vbox" in log:
        fail("Lecture contains an overfull box; inspect the LaTeX log")
    subprocess.run(
        ["latexmk", "-c", "shallow_water_waves_lecture.tex"],
        cwd=LECTURE_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for suffix in (".nav", ".snm", ".vrb"):
        (LECTURE_DIR / f"shallow_water_waves_lecture{suffix}").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="execute all notebooks")
    parser.add_argument("--compile-lecture", action="store_true", help="compile Beamer slides")
    parser.add_argument(
        "--strip-outputs",
        action="store_true",
        help="strip accidental outputs before validating generated notebooks",
    )
    parser.add_argument("--all", action="store_true", help="run structural and full checks")
    args = parser.parse_args()

    if args.strip_outputs:
        strip_notebook_outputs()
    validate_package_version()
    validate_notebooks()
    validate_data()
    validate_lecture_files()
    if args.execute or args.all:
        execute_notebooks()
    if args.compile_lecture or args.all:
        compile_lecture()
    print("Course-material validation passed.")


if __name__ == "__main__":
    main()
