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


def test_part_c_report_template_has_required_sections():
    text = "\n".join(
        cell.source for cell in _read("part_c_project_report_template.ipynb").cells
    )
    for heading in (
        "## Group and research question",
        "## Prediction",
        "## Model scope",
        "## Reusable experiment function",
        "## Experiment",
        "## Results and interpretation",
        "## Conclusion",
        "## Contributions and submission check",
    ):
        assert heading in text
    assert "G (Godkänt)" in text
    assert "U (Underkänt)" in text
    assert "| Criterion | Weight |" not in text
    assert "## Comparison with theory" not in text
    assert "## Limitations" not in text

    report = _read("part_c_project_report_template.ipynb")
    code_cells = [cell for cell in report.cells if cell.cell_type == "code"]
    assert len(code_cells) == 2
    code_text = "\n".join(cell.source for cell in code_cells)
    assert "def make_initial_state(" in code_text
    assert "def run_case(" in code_text
    assert "plot_hovmoller" not in code_text
    assert "animate_case" not in code_text


def test_short_part_b_and_part_c_toolbox_split():
    part_b_text = "\n".join(
        cell.source for cell in _read("part_b_bathymetry_student.ipynb").cells
    )
    toolbox_text = "\n".join(
        cell.source for cell in _read("part_c_project_description.ipynb").cells
    )

    assert "Working time:** about 45 minutes" in part_b_text
    assert "## 1. Build a shelf and make a prediction" in part_b_text
    assert "## 2. Virtual gauges" in part_b_text
    assert "## 3. Take-away" in part_b_text
    assert "part_b_shelf_120m.gif" in part_b_text
    assert "Controlled coastal-depth experiment" not in part_b_text
    assert "uniform_wind_forcing" not in part_b_text
    assert "part_b_shelf_300m.gif" not in part_b_text
    assert "part_b_wind_setup_release.gif" not in part_b_text
    assert "load_bathymetry" not in part_b_text
    assert "make_wind_forcing_from_file" not in part_b_text
    assert "## 4. Setup recipes: copy, adapt, compare" in toolbox_text
    assert "def run_analytic_shelf_case" in toolbox_text
    assert "def run_uniform_wind_case" in toolbox_text
    assert "## 5. Bathymetry supplied as a file" in toolbox_text
    assert "## 6. Wind forcing supplied as a file" in toolbox_text
    assert "ax.contour(" in toolbox_text
    assert "ax.quiver(" in toolbox_text


def test_optional_part_c_setup_recipes_run_when_called():
    notebook = _read("part_c_project_description.ipynb")
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, cell.id, "exec"), namespace)
            if "def run_uniform_wind_case" in cell.source:
                break

    shelf = namespace["run_analytic_shelf_case"](120.0, tmax_hours=0.02)
    wind = namespace["run_uniform_wind_case"](
        wind_off_hours=0.01, tmax_hours=0.02,
    )

    assert np.asarray(shelf["params"].H).shape == (24, 160)
    assert np.asarray(shelf["out"]["eta"]).shape[1:] == (24, 160)
    assert np.asarray(wind["out"]["eta"]).shape[1:] == (24, 160)


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
    assert 16 <= source.read_text(encoding="utf-8").count("\\begin{frame}") <= 28
    assert pdf.stat().st_size > 10_000

    project_source = COURSE / "lecture" / "shallow_water_project_introduction.tex"
    project_pdf = COURSE / "lecture" / "shallow_water_project_introduction.pdf"
    project_text = project_source.read_text(encoding="utf-8")
    assert project_text.count("\\begin{frame}") == 5
    assert r"\documentclass[aspectratio=169]{beamer}" in project_text
    assert r"\usetheme{Madrid}" in project_text
    assert r"\definecolor{deepblue}{RGB}{25,87,140}" in project_text
    assert project_pdf.stat().st_size > 10_000
