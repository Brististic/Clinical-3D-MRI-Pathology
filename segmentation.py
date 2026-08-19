import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from collections import deque

def load_slice_pair(base_path, slice_idx=80):
    flair_vol = nib.load(f"{base_path}_flair.nii.gz").get_fdata(dtype=np.float32)
    seg_vol   = nib.load(f"{base_path}_seg.nii.gz").get_fdata(dtype=np.float32)
    
    flair_slice = flair_vol[:, :, slice_idx]
    seg_slice   = seg_vol[:, :, slice_idx]
    
    # Normalize FLAIR to [0, 255]
    min_val, max_val = np.min(flair_slice), np.max(flair_slice)
    if max_val > min_val:
        flair_norm = ((flair_slice - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        flair_norm = np.zeros_like(flair_slice, dtype=np.uint8)
        
    gt_mask = (seg_slice > 0).astype(np.uint8)
    return flair_norm, gt_mask

# --- 1. Statistical Adaptive Region Growing ---
def statistical_region_growing(img, seed, std_multiplier=1.8, max_iter=25000):
    """
    Grows region based on dynamic running mean and standard deviation of the segmented area.
    """
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
        
        # Calculate dynamic region statistics
        mean_val = np.mean(region_pixels)
        std_val = max(np.std(region_pixels), 8.0) # minimum std floor to prevent stall
        
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                pixel_val = float(img[nr, nc])
                
                # Check if neighbor falls within current adaptive distribution
                if abs(pixel_val - mean_val) <= (std_multiplier * std_val):
                    # Ensure we don't bleed into zero-background
                    if pixel_val > 15:
                        region_pixels.append(pixel_val)
                        queue.append((nr, nc))
                        
    return segmented_mask

# --- 2. Advanced Multi-Step Morphology ---
def refine_morphology(mask):
    """Fills internal holes and cleans edge boundaries."""
    # Elliptical kernel for anatomical biological shapes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Fill internal hypointense cavities
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # Remove boundary whiskers
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Retain only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels > 1:
        # Index 0 is background, find max among remaining
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        opened = (labels == largest_label).astype(np.uint8)
        
    return opened

def compute_metrics(pred_mask, gt_mask):
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    total_pred = np.sum(pred_mask == 1)
    total_gt = np.sum(gt_mask == 1)
    
    if total_pred + total_gt == 0:
        return 1.0, 1.0
    dice = (2.0 * intersection) / (total_pred + total_gt)
    union = total_pred + total_gt - intersection
    iou = intersection / union if union > 0 else 0.0
    return dice, iou

if __name__ == "__main__":
    patient_id = "BraTS2021_00621"
    base_path = f"data/{patient_id}/{patient_id}"
    slice_idx = 80
    
    flair_img, gt_mask = load_slice_pair(base_path, slice_idx=slice_idx)
    
    # Pick a solid hyperintense seed point within the ground truth
    gt_coords = np.argwhere(gt_mask == 1)
    if len(gt_coords) > 0:
        # Sample the brightest pixel inside the ground truth region
        gt_pixel_values = [flair_img[r, c] for r, c in gt_coords]
        best_seed_idx = np.argmax(gt_pixel_values)
        seed_point = tuple(gt_coords[best_seed_idx])
    else:
        seed_point = (120, 120)
        
    print(f"Refined Seed Point: {seed_point} with Intensity: {flair_img[seed_point[0], seed_point[1]]}")
    
    # 1. Run Statistical Region Growing
    raw_seg = statistical_region_growing(flair_img, seed=seed_point, std_multiplier=1.7)
    
    # 2. Refine with Morphological Pipeline
    refined_seg = refine_morphology(raw_seg)
    
    # 3. Calculate Performance Metrics
    dice_score, iou_score = compute_metrics(refined_seg, gt_mask)
    print(f"Upgraded Results -> Dice Score (DSC): {dice_score:.4f} | IoU (Jaccard): {iou_score:.4f}")
    
    # 4. Display Comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(flair_img, cmap="gray")
    axes[0].plot(seed_point[1], seed_point[0], 'go', markersize=7)
    axes[0].set_title(f"FLAIR (Brightest Seed: {seed_point})")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    
    axes[2].imshow(refined_seg, cmap="gray")
    axes[2].set_title("Segmented Output")
    
    # Green = Ground Truth, Red = Prediction, Yellow = Perfect Overlap
    rgb_overlay = np.zeros((*flair_img.shape, 3), dtype=np.uint8)
    rgb_overlay[..., 0] = refined_seg * 255  # Red
    rgb_overlay[..., 1] = gt_mask * 255      # Green
    
    axes[3].imshow(flair_img, cmap="gray")
    axes[3].imshow(rgb_overlay, alpha=0.5)
    axes[3].set_title(f"Overlap Visualizer (DSC: {dice_score:.2f})")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()