# Installation for the shallow-water laboratory

Complete this setup before Part A if possible. The laboratory uses a dedicated
Miniconda environment, Visual Studio Code, and Jupyter notebooks opened directly
inside VS Code.

## 1. Install Miniconda and VS Code

Install both applications before creating the course environment:

- [Miniconda](https://docs.conda.io/miniconda.html): choose the installer for
  your operating system and processor.
- [Visual Studio Code](https://code.visualstudio.com/Download).

In VS Code, open **Extensions** and install these two Microsoft extensions:

- **Python** (`ms-python.python`)
- **Jupyter** (`ms-toolsai.jupyter`)

The Jupyter extension lets VS Code open an `.ipynb` file as a notebook, run its
cells, and display figures and animations below the cells.

### Windows

1. Run the Miniconda graphical installer for your account. The default options
   are suitable; Miniconda does not need to be added to the system `PATH`.
2. Open **Miniconda Prompt** from the Start menu for the commands below.
3. Install VS Code with the user installer if it is not already managed by your
   institution.

### macOS

1. Check **Apple menu > About This Mac** to determine whether the computer uses
   Apple silicon or an Intel processor, then choose the matching Miniconda
   installer.
2. Run the graphical `.pkg` installer and open a new Terminal window afterwards.
3. Install the matching Apple-silicon or Intel build of VS Code.

If `conda` is not found in a new Terminal, open the **Miniconda** application or
run the shell-initialization command shown by the installer, then close and
reopen Terminal.

### Linux

1. Download the Miniconda shell installer matching the processor architecture.
2. Run it from a terminal, accept initialization when prompted, then close and
   reopen the terminal.
3. Install VS Code using the method approved for your Linux distribution.

If `conda` is still not found, use the Miniconda installation guide for your
distribution or ask the instructor rather than installing a second Python.

## 2. Create the course environment

Use **Miniconda Prompt** on Windows or a terminal on macOS/Linux. Run each line
separately:

```text
conda create --name shallowwater-lab python=3.11 -y
conda activate shallowwater-lab
python -m pip install --upgrade pip
python -m pip install "shallowwater==0.1.4" ipykernel
```

The base package includes NumPy, Matplotlib, Jupyter, and Pillow, which is used
to write the GIF animations. To add optional numba acceleration, run:

```text
python -m pip install "shallowwater[numba]==0.1.4"
```

Numba is not required. The first numba-backed model run can be slower while the
numerical functions are compiled.

## 3. Open the laboratory in VS Code

1. Download and unpack the complete `MT1562_python_lab_waves` folder. Keep its
   `notebooks`, `data`, and `animations` folders together.
2. In VS Code, choose **File > Open Folder...** and select
   `MT1562_python_lab_waves`. If VS Code asks whether you trust the authors of
   the folder, confirm only for the course copy supplied by the instructor.
3. Open `notebooks/part_a_waves_student.ipynb` from the Explorer.
4. Click **Select Kernel** in the upper-right corner of the notebook. Choose
   **Python Environments**, then `shallowwater-lab (Python 3.11)`.
5. Run the first code cell with the triangular Run button. `Shift+Enter` also
   runs the current cell and moves to the next one.

VS Code may show both an interpreter in its status bar and a notebook kernel in
the upper-right corner. For this course, the notebook kernel must be
`shallowwater-lab`.

## 4. Verify the installation

Run this in a notebook cell:

```python
import shallowwater
from matplotlib.animation import writers

print(shallowwater.backend_info())
print("GIF writer available:", writers.is_available("pillow"))
```

The package version should be `0.1.4`, and the GIF-writer check should print
`True`. The backend may be NumPy or numba.

## Troubleshooting

### The environment does not appear in the kernel list

1. Close and reopen the notebook.
2. Click **Select Kernel > Select Another Kernel... > Python Environments**.
3. In a terminal with `shallowwater-lab` activated, run:

   ```text
   python -m pip install --upgrade ipykernel
   ```

4. In VS Code, open the Command Palette and run **Developer: Reload Window**.

### VS Code uses the wrong Python

Use **Python: Select Interpreter** from the Command Palette and select
`shallowwater-lab`, then independently select the same environment with the
notebook's **Select Kernel** control. Restart the notebook kernel after changing
it.

### `No module named shallowwater`

The package was installed into a different Python environment. Activate the
course environment and reinstall with the interpreter-specific form:

```text
conda activate shallowwater-lab
python -m pip install "shallowwater==0.1.4"
```

Then select `shallowwater-lab` as the notebook kernel and restart it.

### Numba fails or gives a long first-run delay

Continue with the base NumPy backend. Close VS Code, activate the environment,
set `SHALLOWWATER_USE_NUMBA=0`, and launch VS Code from the same prompt.

On Windows in Miniconda Prompt:

```text
set SHALLOWWATER_USE_NUMBA=0
code .
```

On macOS or Linux:

```text
export SHALLOWWATER_USE_NUMBA=0
code .
```

If `code` is not available as a terminal command, skip numba installation and
use the base environment, or ask the instructor for help. Numerical results
should agree within floating-point tolerance.

### An animation does not appear

First confirm that all earlier cells ran and that `GIF writer available` is
`True`. The inline player can take a few seconds to render. The notebook also
saves a `.gif` copy under `animations/`; open that file from the VS Code Explorer
if the inline player is not displayed.

### The Part C toolbox cannot find a data file

Open the complete `MT1562_python_lab_waves` folder in VS Code. Do not move a
notebook out of `notebooks/` or separate it from the sibling `data/` folder.
