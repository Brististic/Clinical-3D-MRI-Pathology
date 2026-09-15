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

Create a small synthetic study for local UI testing:

```bash
python scripts/create_demo_data.py
```

This creates `data/BraTS2021_00621/` with synthetic FLAIR and segmentation files
matching the dashboard's expected paths. The generated data is not derived from
patients and must not be used for clinical validation.

Run the test suite:

```bash
python -m pytest -q
```

## Using an authorized patient study

The dashboard also supports local NIfTI uploads, so patient data does not need
to be committed to GitHub. Start the dashboard, select **Upload NIfTI files**,
and provide a 3D FLAIR `.nii`/`.nii.gz` file. A matching segmentation label is
optional; it enables Dice, IoU, and reference-guided volume analysis.

Uploaded FLAIR and label volumes must have the same 3D dimensions. The
application validates this before processing. Do not upload patient data to
untrusted environments, and do not treat the prototype's output as a clinical
diagnosis.

Run the dashboard:

```bash
streamlit run dashboard.py
```

## CI

The repository includes a GitHub Actions workflow under `.github/workflows/pytest.yml` that runs the test suite on every push and pull request.
