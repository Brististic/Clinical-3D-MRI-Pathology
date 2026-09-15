# NeuroDelineate: Clinical 3D MRI Pathology Segmentation & Diagnostic Suite

This project provides a classical image-processing workflow for 3D MRI pathology segmentation, volumetric analysis, and automated report generation.

## Features

- 3D MRI slice traversal and intensity normalization
- Statistical region growing with morphological cleanup
- GrabCut-based foreground/background segmentation
- Volume and lesion metric calculations
- Clinical PDF report generation
- Streamlit-based diagnostic dashboard

## Project structure

- `src/` contains reusable segmentation and IO logic
- `dashboard.py` launches the interactive visual diagnostic interface
- `test_graph_cut.py` contains the original prototype comparison script
- `tests/` contains a real pytest suite for validation

## Local setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest -q
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

## CI

The repository includes a GitHub Actions workflow under `.github/workflows/pytest.yml` that runs the test suite on every push and pull request.
