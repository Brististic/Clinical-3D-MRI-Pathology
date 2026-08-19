# NeuroDelineate: Clinical 3D MRI Pathology Segmentation & Diagnostic Suite

An end-to-end medical image processing platform designed for automated lesion delineation, volumetric quantification, and standardized diagnostic report generation from 3D multi-modal MRI scans (BraTS Dataset). Built entirely using classical Digital Image Processing (DIP) algorithms and Python.

---

## Key Features

* **3D Volumetric Traversal:** Real-time slice scrubbing across 155-slice axial brain scans with dynamic intensity windowing and pseudo-color thermal heatmaps.
* **Classical Segmentation Engine:** Automated statistical homogeneity region growing with adaptive thresholding ($k \cdot \sigma$) and morphological post-processing (closing, opening, and connected component filtering).
* **Multi-Domain Preprocessing:** Spatial-domain Contrast Limited Adaptive Histogram Equalization (CLAHE) and Frequency-Domain 2D Discrete Fourier Transform (2D-DFT) Gaussian high-pass filtering for edge enhancement.
* **Quantitative Validation:** Automated slice-level and volume-level evaluation against expert radiologist ground truth using Dice Similarity Coefficient (DSC) and Jaccard Index (IoU).
* **True Volumetric Calculation:** Converts pixel voxel volumes into clinical metric units ($\text{cm}^3$) using NIfTI coordinate affine matrices.
* **Automated Clinical PDF Export:** Generates signed, hospital-ready diagnostic summary reports containing embedded patient metadata, quantitative findings, and multi-panel pathology overlays via ReportLab.

---

## Mathematical Formulations

### 1. Statistical Homogeneity Criterion
A candidate pixel $(r, c)$ is merged into the region $R$ if:
$$\vert{}I(r, c) - \mu_R\vert{} \le k \cdot \sigma_R$$
Where $\mu_R$ and $\sigma_R$ denote the running mean and standard deviation of the segmented region, and $k$ is the sensitivity multiplier.

### 2. Validation Metrics
* **Dice Similarity Coefficient (DSC):**
  $$\text{DSC} = \frac{2 \vert{}A \cap B\vert{}}{\vert{}A\vert{} + \vert{}B\vert{}}$$
* **Jaccard Index (IoU):**
  $$\text{IoU} = \frac{\vert{}A \cap B\vert{}}{\vert{}A \cup B\vert{}}$$

### 3. Physical Volumetric Integration
$$\text{Volume } (\text{cm}^3) = \frac{N_{\text{voxels}} \times (v_x \times v_y \times v_z)}{1000}$$
Where $v_x, v_y, v_z$ represent the spatial voxel dimensions in millimeters extracted from the NIfTI header.

---

## Project Structure

```text
medical-segmentation-suite/
│
├── data/                               <-- NIfTI BraTS patient scans (.nii.gz)
│   └── BraTS2021_00621/
│
├── src/
│   ├── io/
│   │   ├── data_loader.py              <-- 3D volume parser & intensity normalizer
│   │   └── report_generator.py         <-- In-memory clinical PDF engine
│   ├── filters/
│   │   └── enhancement.py              <-- 2D-DFT & CLAHE implementations
│   ├── segmentation/
│   │   ├── region_growing.py           <-- Statistical region growing algorithm
│   │   └── graph_cut.py                <-- OpenCV Graph-Cut energy minimization
│   ├── metrics/
│   │   └── evaluation.py               <-- DSC, IoU, and Hausdorff distance
│   └── compression/
│       └── codecs.py                   <-- Huffman lossless & DCT lossy codecs
│
├── dashboard.py                        <-- Interactive Streamlit clinical interface
├── requirements.txt                    <-- Environment dependencies
├── .gitignore
└── README.md

Quickstart & Installation
1. Clone the Repository
Bash
git clone [https://github.com/Brististic/Clinical-3D-MRI-Pathology.git]
cd medical-segmentation-suite

2. Set Up Virtual Environment
Bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Bash
pip install -r requirements.txt

4. Launch the Clinical Diagnostic Dashboard
Bash
streamlit run dashboard.py