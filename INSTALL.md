# Installation

## Requirements

- Windows, Linux, or macOS
- Python 3.11
- Git
- Git LFS
- A CPU or CUDA installation supported by the selected PyTorch build

The reference environment was tested with Python 3.11.15, PyTorch 2.8.0+cu126, and CUDA 12.6 on Windows. CPU execution is supported but can be slower.

## Clone the repository

Install and enable Git LFS before cloning so the model checkpoints are downloaded:

```bash
git lfs install
git clone https://github.com/aaronwzc123-rgb/interpretable-world-model.git
cd interpretable-world-model
```

If the repository was cloned before Git LFS was installed, run:

```bash
git lfs pull
```

## Create the Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install PyTorch

Install a PyTorch build appropriate for the machine. For a CPU-only setup:

```bash
python -m pip install torch
```

For CUDA, install the build recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

Do not install a second PyTorch build after selecting a CUDA-specific one.

## Verify the installation

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run the regression tests:

```bash
python -m unittest tests.test_regressions
```

The expected result is 12 passing tests.

## Run the demos

Interactive visualisation:

```bash
python interactive_demo.py
```

On Windows, the launcher can also be used:

```bat
run_interactive_demo.bat
```

Acceptance notebooks:

```bash
python -m notebook
```

On Windows, double-click `run_acceptance_notebooks.bat`.

## Troubleshooting

If an existing virtual environment reports that pip is missing, repair it with:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

If checkpoint files appear as small text pointer files, Git LFS was not installed or the objects were not downloaded. Run:

```bash
git lfs install
git lfs pull
```

If CUDA is unavailable, the project can still run with `--device cpu`; evaluation will usually be slower.
