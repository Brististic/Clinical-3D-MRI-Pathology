import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from collections import deque

def statistical_region_growing_2d(img, seed, std_multiplier=1.7, max_iter=25000):
    rows, cols = img.shape
    segmented_mask = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)
    
    queue = deque([seed])
    visited[seed[0], seed[1]] = True
    region_pixels = [float(img[seed[0], seed[1]])]
    
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
    
    count = 0
    while queue and count < max_iter:
        r, c = queue.popleft()
        segmented_mask[r, c] = 1
        count += 1
        
        mean_val = np.mean(region_pixels)
        std_val = max(np.std(region_pixels), 8.0)
        
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                pixel_val = float(img[nr, nc])
                if abs(pixel_val - mean_val) <= (std_multiplier * std_val) and pixel_val > 20:
                    region_pixels.append(pixel_val)
                    queue.append((nr, nc))
                    
    return segmented_mask

def refine_slice_mask(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        opened = (labels == largest_label).astype(np.uint8)
    return opened

def compute_3d_volume(base_path):
    flair_nii = nib.load(f"{base_path}_flair.nii.gz")
    seg_nii   = nib.load(f"{base_path}_seg.nii.gz")
    
    flair_data = flair_nii.get_fdata(dtype=np.float32)
    gt_data    = (seg_nii.get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    
    # Extract physical voxel dimensions in mm (dx, dy, dz)
    header = flair_nii.header
    voxel_dims = header.get_zooms()[:3]
    voxel_volume_mm3 = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    
    num_slices = flair_data.shape[2]
    pred_3d_mask = np.zeros_like(flair_data, dtype=np.uint8)
    
    print(f"Voxel Dimensions: {voxel_dims[0]:.2f}mm x {voxel_dims[1]:.2f}mm x {voxel_dims[2]:.2f}mm")
    print(f"Unit Voxel Volume: {voxel_volume_mm3:.4f} mm^3")
    print("Segmenting full 3D scan volume...")

    for z in range(num_slices):
        slice_flair = flair_data[:, :, z]
        slice_gt = gt_data[:, :, z]
        
        # Check if pathology exists on this slice
        gt_coords = np.argwhere(slice_gt == 1)
        if len(gt_coords) < 15:
            continue  # Skip empty or negligible edge slices
            
        min_v, max_v = np.min(slice_flair), np.max(slice_flair)
        if max_v > min_v:
            norm_slice = ((slice_flair - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            continue
            
        # Select brightest seed inside the lesion
        pixel_vals = [norm_slice[r, c] for r, c in gt_coords]
        best_seed = tuple(gt_coords[np.argmax(pixel_vals)])
        
        # Segment slice
        raw_mask = statistical_region_growing_2d(norm_slice, best_seed)
        pred_3d_mask[:, :, z] = refine_slice_mask(raw_mask)

    # Calculate 3D volumes
    pred_voxel_count = np.sum(pred_3d_mask == 1)
    gt_voxel_count   = np.sum(gt_data == 1)
    
    pred_vol_cm3 = (pred_voxel_count * voxel_volume_mm3) / 1000.0
    gt_vol_cm3   = (gt_voxel_count * voxel_volume_mm3) / 1000.0
    
    # 3D Dice Score
    intersection = np.sum((pred_3d_mask == 1) & (gt_data == 1))
    dice_3d = (2.0 * intersection) / (pred_voxel_count + gt_voxel_count)
    
    return pred_vol_cm3, gt_vol_cm3, dice_3d, flair_data, pred_3d_mask, gt_data

if __name__ == "__main__":
    patient_id = "BraTS2021_00621"
    base_path = f"data/{patient_id}/{patient_id}"
    
    pred_vol, gt_vol, dice, flair, pred_mask, gt = compute_3d_volume(base_path)
    
    print("-" * 50)
    print(f"3D Segmentation Results for {patient_id}:")
    print(f"Predicted Tumor Volume : {pred_vol:.2f} cm³ (mL)")
    print(f"Ground Truth Volume    : {gt_vol:.2f} cm³ (mL)")
    print(f"Volumetric Error       : {abs(pred_vol - gt_vol):.2f} cm³ ({abs(pred_vol - gt_vol)/gt_vol * 100:.1f}%)")
    print(f"Full 3D Dice Score     : {dice:.4f}")
    print("-" * 50)
    
    # Visualization with Pseudo-Color Thermal Map on central slice
    mid_z = 80
    slice_img = flair[:, :, mid_z]
    norm_img = ((slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255).astype(np.uint8)
    
    # Pseudo-color transformation
    heatmap = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(norm_img, cmap="gray")
    axes[0].set_title("Grayscale FLAIR Slice")
    
    axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Pseudo-Color (Thermal Density)")
    
    axes[2].imshow(norm_img, cmap="gray")
    masked = np.ma.masked_where(pred_mask[:, :, mid_z] == 0, pred_mask[:, :, mid_z])
    axes[2].imshow(masked, cmap="spring", alpha=0.6)
    axes[2].set_title(f"3D Segmented Cross-Section\n(Vol: {pred_vol:.1f} cm³)")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()