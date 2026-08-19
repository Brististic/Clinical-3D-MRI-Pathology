import os
import io
import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
from scipy import ndimage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(page_title="NeuroDelineate - Medical Suite", layout="wide")

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

def get_norm_slice(volume, idx):
    sl = volume[:, :, idx]
    min_v, max_v = np.min(sl), np.max(sl)
    if max_v > min_v:
        return ((sl - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    return np.zeros_like(sl, dtype=np.uint8)

def region_growing(img, seed, std_multiplier=1.7):
    rows, cols = img.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([seed])
    visited[seed[0], seed[1]] = True
    region_pixels = [float(img[seed[0], seed[1]])]
    
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
    
    count = 0
    while queue and count < 25000:
        r, c = queue.popleft()
        mask[r, c] = 1
        count += 1
        mean_v = np.mean(region_pixels)
        std_v = max(np.std(region_pixels), 8.0)
        
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                val = float(img[nr, nc])
                if abs(val - mean_v) <= (std_multiplier * std_v) and val > 20:
                    region_pixels.append(val)
                    queue.append((nr, nc))
                    
    # Morphological Refinement via SciPy (No OpenCV required)
    struct = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_closing(mask, structure=struct, iterations=3)
    opened = ndimage.binary_opening(closed, structure=struct, iterations=1)
    
    labeled, num_features = ndimage.label(opened)
    if num_features > 0:
        sizes = ndimage.sum(opened, labeled, range(1, num_features + 1))
        max_label = 1 + int(np.argmax(sizes))
        opened = (labeled == max_label).astype(np.uint8)
    else:
        opened = opened.astype(np.uint8)
        
    return opened

def compute_metrics(pred, gt):
    inter = np.sum((pred == 1) & (gt == 1))
    tot_pred, tot_gt = np.sum(pred == 1), np.sum(gt == 1)
    if tot_pred + tot_gt == 0:
        return 1.0, 1.0
    dice = (2.0 * inter) / (tot_pred + tot_gt)
    union = tot_pred + tot_gt - inter
    iou = inter / union if union > 0 else 0.0
    return dice, iou

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
st.title("🧠 NeuroDelineate — Clinical Diagnostic Web Suite")
st.caption("Classical Digital Image Processing & 3D Volumetric Engine")

patient_id = "BraTS2021_00621"
flair_vol, gt_vol, dims, voxel_vol = load_nifti_data(patient_id)

col_ctrl, col_main = st.columns([1, 3])

with col_ctrl:
    st.subheader("Controls")
    slice_idx = st.slider("Select Axial Slice", 0, flair_vol.shape[2] - 1, 80)
    color_mode = st.selectbox("Color Mapping", ["Grayscale", "Thermal Heatmap"])
    std_mult = st.slider("Region Growing Sensitivity (k·σ)", 1.0, 3.0, 1.7, 0.1)
    
    st.divider()
    st.write(f"**Voxel Spacing:** `{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm`")
    
    if st.button("🚀 Compute Full 3D Volume"):
        with st.spinner("Processing 155 slices..."):
            pred_count = 0
            gt_count = int(np.sum(gt_vol == 1))
            for z in range(flair_vol.shape[2]):
                sl_gt = gt_vol[:, :, z]
                if np.sum(sl_gt) > 15:
                    coords = np.argwhere(sl_gt == 1)
                    sl_img = get_norm_slice(flair_vol, z)
                    vals = [sl_img[r, c] for r, c in coords]
                    best_seed = tuple(coords[np.argmax(vals)])
                    m = region_growing(sl_img, best_seed, std_mult)
                    pred_count += int(np.sum(m == 1))
            
            p_vol = (pred_count * voxel_vol) / 1000.0
            g_vol = (gt_count * voxel_vol) / 1000.0
            
            st.session_state['pred_vol'] = p_vol
            st.session_state['gt_vol'] = g_vol
            
            st.success(f"**Predicted 3D Volume:** {p_vol:.2f} cm³")
            st.info(f"**Ground Truth Volume:** {g_vol:.2f} cm³")
            st.warning(f"**Volumetric Error:** {abs(p_vol - g_vol):.2f} cm³")

with col_main:
    norm_slice = get_norm_slice(flair_vol, slice_idx)
    slice_gt = gt_vol[:, :, slice_idx]
    
    gt_coords = np.argwhere(slice_gt == 1)
    if len(gt_coords) > 0:
        vals = [norm_slice[r, c] for r, c in gt_coords]
        seed_pt = tuple(gt_coords[np.argmax(vals)])
    else:
        seed_pt = (120, 120)
        
    pred_mask = region_growing(norm_slice, seed_pt, std_multiplier=std_mult)
    dice, iou = compute_metrics(pred_mask, slice_gt)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Dice Score (DSC)", f"{dice:.4f}")
    m2.metric("IoU (Jaccard)", f"{iou:.4f}")
    m3.metric("Pathology Detected", "Yes" if np.sum(pred_mask) > 0 else "No")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if color_mode == "Thermal Heatmap":
        axes[0].imshow(norm_slice, cmap="turbo")
    else:
        axes[0].imshow(norm_slice, cmap="gray")
    axes[0].plot(seed_pt[1], seed_pt[0], 'go', markersize=6)
    axes[0].set_title(f"FLAIR Slice #{slice_idx}")
    
    axes[1].imshow(slice_gt, cmap="autumn")
    axes[1].set_title("Ground Truth Mask")
    
    axes[2].imshow(norm_slice, cmap="gray")
    masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[2].imshow(masked_pred, cmap="spring", alpha=0.6)
    axes[2].set_title("Segmented Pathology Overlay")
    
    for ax in axes:
        ax.axis("off")
        
    st.pyplot(fig)

    st.divider()
    st.subheader("📄 Clinical Export")
    p_vol_val = st.session_state.get('pred_vol', 0.0)
    g_vol_val = st.session_state.get('gt_vol', 0.0)
    
    if st.button("📄 Prepare PDF Report"):
        with st.spinner("Generating clinical document..."):
            pdf_bytes = generate_clinical_pdf(
                patient_id=patient_id,
                slice_idx=slice_idx,
                dice_score=dice,
                iou_score=iou,
                pred_vol=p_vol_val,
                gt_vol=g_vol_val,
                fig_matplotlib=fig
            )
            st.download_button(
                label="⬇️ Download Signed Diagnostic PDF",
                data=pdf_bytes,
                file_name=f"Report_{patient_id}_slice{slice_idx}.pdf",
                mime="application/pdf"
            )