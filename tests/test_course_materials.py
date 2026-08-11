from pathlib import Path
import re

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "MT1562_python_lab_waves"
NOTEBOOKS = COURSE / "notebooks"


def _read(name):
    return nbformat.read(NOTEBOOKS / name, as_version=4)


def test_generated_notebooks_are_clean_and_code_compiles():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        ids = [cell.id for cell in notebook.cells]
        assert len(ids) == len(set(ids))
        for index, cell in enumerate(notebook.cells):
            assert "/Users/" not in cell.source
            assert not re.search(r"[A-Za-z]:\\Users\\", cell.source)
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:{index}", "exec")
                assert cell.execution_count is None
                assert cell.outputs == []


def test_student_solution_pairs_share_structure_and_code():
    for stem in ("part_a_waves", "part_b_bathymetry"):
        student = _read(f"{stem}_student.ipynb")
        solution = _read(f"{stem}_solutions.ipynb")
        assert [cell.id for cell in student.cells] == [cell.id for cell in solution.cells]
        for student_cell, solution_cell in zip(student.cells, solution.cells):
            assert student_cell.cell_type == solution_cell.cell_type
            if student_cell.cell_type == "code":
                assert student_cell.source == solution_cell.source
        assert "**Solution.**" not in "\n".join(cell.source for cell in student.cells)
        assert "**Solution.**" in "\n".join(cell.source for cell in solution.cells)


def test_part_c_has_required_report_sections():
    text = "\n".join(cell.source for cell in _read("part_c_project.ipynb").cells)
    for heading in (
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
    ):
        assert heading in text


def test_course_input_data_are_valid():
    with np.load(COURSE / "data" / "example_bathymetry.npz", allow_pickle=False) as data:
        assert data["H"].shape == (24, 160)
        assert np.isfinite(data["H"]).all()
        assert np.min(data["H"]) > 0
    with np.load(COURSE / "data" / "example_wind_forcing.npz", allow_pickle=False) as data:
        assert data["tau_x"].shape == (24, 160)
        assert data["tau_y"].shape == (24, 160)


def test_lecture_source_and_pdf_exist():
    source = COURSE / "lecture" / "shallow_water_waves_lecture.tex"
    pdf = COURSE / "lecture" / "shallow_water_waves_lecture.pdf"
    assert 20 <= source.read_text(encoding="utf-8").count("\\begin{frame}") <= 28
    assert pdf.stat().st_size > 10_000
