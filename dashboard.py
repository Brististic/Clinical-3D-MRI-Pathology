import os
import io
import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
from scipy import ndimage
from src.io.data_loader import (
    get_normalized_slice,
    load_nifti_bytes,
    validate_volume_pair,
)
from src.metrics.evaluation import compute_metrics as compute_mask_metrics
from src.segmentation.region_growing import (
    refine_mask,
    statistical_region_growing,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="NeuroDelineate - Medical Suite", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(135deg, #0f172a, #164e63); color: white; margin-bottom: 1.25rem; }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { color: #cbd5e1; margin: .35rem 0 0; }
    div[data-testid="stMetric"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0;
        padding: .75rem;
        border-radius: 12px;
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] p {
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper Functions ---
@st.cache_resource
def load_nifti_data(patient_id):
    if os.path.exists(f"data/{patient_id}"):
        base_path = f"data/{patient_id}/{patient_id}"
    else:
        base_path = f"sample_data/{patient_id}/{patient_id}"

    flair_nii = nib.load(f"{base_path}_flair.nii.gz")
    seg_nii = nib.load(f"{base_path}_seg.nii.gz")
    
    flair_data = flair_nii.get_fdata(dtype=np.float32)
    gt_data = (seg_nii.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    
    voxel_dims = flair_nii.header.get_zooms()[:3]
    voxel_vol = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
    
    return flair_data, gt_data, voxel_dims, voxel_vol


def load_uploaded_data(flair_file, segmentation_file=None):
    flair_data, flair_header = load_nifti_bytes(flair_file)
    gt_data = None
    if segmentation_file is not None:
        gt_data, _ = load_nifti_bytes(segmentation_file)
        validate_volume_pair(flair_data, gt_data)
        gt_data = (gt_data > 0).astype(np.uint8)
    if flair_data.ndim != 3:
        raise ValueError("The uploaded FLAIR file must contain a 3D volume.")
    voxel_dims = flair_header.get_zooms()[:3]
    voxel_vol = float(np.prod(voxel_dims))
    return flair_data, gt_data, voxel_dims, voxel_vol


def get_norm_slice(volume, idx):
    return get_normalized_slice(volume, idx)

def region_growing(img, seed, std_multiplier=1.7):
    return refine_mask(
        statistical_region_growing(img, seed, std_multiplier=std_multiplier)
    )

def compute_metrics(pred, gt):
    return compute_mask_metrics(pred, gt)

def generate_clinical_pdf(patient_id, slice_idx, dice_score, iou_score, pred_vol, gt_vol, fig_matplotlib):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#0f172a'))
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=4)
    normal_text = ParagraphStyle('BodyDark', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#334155'))

    story = [
        Paragraph("NEURODELINEATE — CLINICAL DIAGNOSTIC REPORT", title_style),
        Paragraph("Automated Volumetric Brain Pathology Segmentation Suite", normal_text),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceBefore=6, spaceAfter=10)
    ]

    meta_data = [
        [Paragraph("<b>Patient Identifier:</b>", normal_text), Paragraph(patient_id, normal_text),
         Paragraph("<b>Modality:</b>", normal_text), Paragraph("MRI (FLAIR)", normal_text)],
        [Paragraph("<b>Key Slice Index:</b>", normal_text), Paragraph(f"Slice #{slice_idx}", normal_text),
         Paragraph("<b>Status:</b>", normal_text), Paragraph("Automated Extraction", normal_text)]
    ]
    meta_table = Table(meta_data, colWidths=[120, 145, 110, 155])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Quantitative Volumetric Findings", section_heading))
    vol_err = abs(pred_vol - gt_vol) if gt_vol > 0 else 0.0
    metrics_data = [
        ["Diagnostic Parameter", "Measured Value", "Reference (Ground Truth)", "Clinical Status"],
        ["Predicted Tumor Volume", f"{pred_vol:.2f} cm³", f"{gt_vol:.2f} cm³", f"Error: {vol_err:.2f} cm³"],
        ["Dice Similarity (DSC)", f"{dice_score:.4f}", "1.0000", "High Concordance" if dice_score > 0.7 else "Review Required"],
        ["Jaccard Index (IoU)", f"{iou_score:.4f}", "1.0000", "Concordant" if iou_score > 0.6 else "Review Required"],
    ]
    metrics_table = Table(metrics_data, colWidths=[160, 110, 130, 130])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visual Pathology Delineation", section_heading))
    img_buf = io.BytesIO()
    fig_matplotlib.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=530, height=170))
    story.append(Spacer(1, 10))

    disclaimer = "<b>Disclaimer:</b> Experimental prototype for computer-aided DIP verification."
    story.append(Paragraph(disclaimer, normal_text))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- App UI ---
st.markdown(
    """
    <div class="hero">
        <h1>NeuroDelineate</h1>
        <p>Clinical MRI review workspace for classical segmentation and volumetric validation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Review setup")
    source = st.radio("Study source", ["Local demo study", "Upload NIfTI files"])
    patient_id = "BraTS2021_00621"
    flair_upload = None
    segmentation_upload = None
    if source == "Upload NIfTI files":
        patient_id = st.text_input("Patient identifier", "Uploaded study")
        flair_upload = st.file_uploader("FLAIR image (.nii/.nii.gz)", type=["nii", "gz"])
        segmentation_upload = st.file_uploader(
            "Segmentation label (optional)", type=["nii", "gz"]
        )
    else:
        st.caption("Current study")
        st.text_input("Patient identifier", patient_id, disabled=True)
    try:
        if source == "Upload NIfTI files":
            if flair_upload is None:
                st.info("Upload a FLAIR volume to begin.")
                st.stop()
            flair_vol, gt_vol, dims, voxel_vol = load_uploaded_data(
                flair_upload, segmentation_upload
            )
        else:
            flair_vol, gt_vol, dims, voxel_vol = load_nifti_data(patient_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error("The study could not be loaded. Check that the NIfTI files are valid and compatible.")
        st.exception(exc)
        st.stop()

    slice_idx = st.slider("Axial slice", 0, flair_vol.shape[2] - 1, min(80, flair_vol.shape[2] - 1))
    color_mode = st.radio("Image display", ["Grayscale", "Thermal Heatmap"], horizontal=False)
    std_mult = st.slider("Region-growing sensitivity (k·σ)", 1.0, 3.0, 1.7, 0.1)
    st.divider()
    st.caption("Acquisition")
    st.write(f"**Volume:** `{flair_vol.shape[0]} × {flair_vol.shape[1]} × {flair_vol.shape[2]}`")
    st.write(f"**Voxel spacing:** `{dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm`")
    st.caption("Prototype output should be reviewed by a qualified clinician.")

norm_slice = get_norm_slice(flair_vol, slice_idx)
has_reference = gt_vol is not None
slice_gt = gt_vol[:, :, slice_idx] if has_reference else np.zeros_like(norm_slice)
gt_coords = np.argwhere(slice_gt == 1)
if len(gt_coords) > 0:
    vals = [norm_slice[r, c] for r, c in gt_coords]
    seed_pt = tuple(gt_coords[np.argmax(vals)])
else:
    seed_pt = (norm_slice.shape[0] // 2, norm_slice.shape[1] // 2)

with st.spinner("Segmenting selected slice..."):
    pred_mask = region_growing(norm_slice, seed_pt, std_multiplier=std_mult)
dice, iou = compute_metrics(pred_mask, slice_gt) if has_reference else (None, None)

metric_cols = st.columns(4)
metric_cols[0].metric("Dice similarity", f"{dice:.4f}" if dice is not None else "N/A")
metric_cols[1].metric("Jaccard / IoU", f"{iou:.4f}" if iou is not None else "N/A")
metric_cols[2].metric("Predicted area", f"{int(pred_mask.sum()):,} px")
metric_cols[3].metric("Selected slice", f"{slice_idx} / {flair_vol.shape[2] - 1}")

st.subheader("Slice review")
st.caption("Compare the source image, reference annotation, and algorithm output. The green marker shows the automatically selected seed.")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axes[0].imshow(norm_slice, cmap="turbo" if color_mode == "Thermal Heatmap" else "gray")
axes[0].plot(seed_pt[1], seed_pt[0], "go", markersize=6)
axes[0].set_title("FLAIR source")
axes[1].imshow(slice_gt, cmap="autumn")
axes[1].set_title("Reference mask")
axes[2].imshow(norm_slice, cmap="gray")
axes[2].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="spring", alpha=0.65)
axes[2].set_title("Predicted overlay")
for ax in axes:
    ax.axis("off")
st.pyplot(fig, use_container_width=True)

st.divider()
volume_tab, export_tab = st.tabs(["3D volume analysis", "Clinical export"])

with volume_tab:
    st.subheader("Full-volume analysis")
    st.caption("Runs the same region-growing workflow across slices containing reference pathology.")
    if not has_reference:
        st.info("Upload a segmentation label to enable reference-guided full-volume analysis.")
    elif st.button("Compute full 3D volume", type="primary"):
        with st.spinner(f"Processing {flair_vol.shape[2]} slices..."):
            pred_count = 0
            gt_count = int(np.sum(gt_vol == 1))
            for z in range(flair_vol.shape[2]):
                sl_gt = gt_vol[:, :, z]
                if np.sum(sl_gt) > 15:
                    coords = np.argwhere(sl_gt == 1)
                    sl_img = get_norm_slice(flair_vol, z)
                    vals = [sl_img[r, c] for r, c in coords]
                    best_seed = tuple(coords[np.argmax(vals)])
                    pred_count += int(np.sum(region_growing(sl_img, best_seed, std_mult) == 1))
            p_vol = (pred_count * voxel_vol) / 1000.0
            g_vol = (gt_count * voxel_vol) / 1000.0
            st.session_state["pred_vol"] = p_vol
            st.session_state["gt_vol"] = g_vol
    if "pred_vol" in st.session_state:
        v1, v2, v3 = st.columns(3)
        v1.metric("Predicted volume", f"{st.session_state['pred_vol']:.2f} cm³")
        v2.metric("Reference volume", f"{st.session_state['gt_vol']:.2f} cm³")
        v3.metric("Absolute difference", f"{abs(st.session_state['pred_vol'] - st.session_state['gt_vol']):.2f} cm³")
    else:
        st.info("Run the volume analysis to populate the 3D findings.")

with export_tab:
    st.subheader("Clinical export")
    st.caption("Generate a PDF snapshot of the selected slice and the latest volume findings.")
    p_vol_val = st.session_state.get("pred_vol", 0.0)
    g_vol_val = st.session_state.get("gt_vol", 0.0)
    if st.button("Prepare PDF report"):
        with st.spinner("Generating clinical document..."):
            pdf_bytes = generate_clinical_pdf(
                patient_id,
                slice_idx,
                dice or 0.0,
                iou or 0.0,
                p_vol_val,
                g_vol_val,
                fig,
            )
            st.download_button(
                "Download diagnostic PDF",
                data=pdf_bytes,
                file_name=f"Report_{patient_id}_slice{slice_idx}.pdf",
                mime="application/pdf",
                type="primary",
            )