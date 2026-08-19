import numpy as np
import matplotlib.pyplot as plt
from src.io.data_loader import load_volume, get_normalized_slice
from src.segmentation.region_growing import statistical_region_growing, refine_mask
from src.segmentation.graph_cut import graph_cut_segmentation

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
    
    # 1. Load scan and ground truth
    flair_vol, _ = load_volume(f"{base_path}_flair.nii.gz")
    seg_vol, _   = load_volume(f"{base_path}_seg.nii.gz")
    
    slice_img = get_normalized_slice(flair_vol, slice_idx)
    gt_mask   = (seg_vol[:, :, slice_idx] > 0).astype(np.uint8)
    
    # 2. Select Seeds
    gt_coords = np.argwhere(gt_mask == 1)
    if len(gt_coords) > 0:
        # Sample points inside ground truth for tumor seeds
        fg_seeds = [tuple(gt_coords[len(gt_coords) // 2]), tuple(gt_coords[0])]
    else:
        fg_seeds = [(120, 120)]
        
    bg_seeds = [(30, 30), (120, 180), (200, 200)]  # Normal tissue / background samples
    
    # 3. Run Statistical Region Growing
    print("Running Region Growing...")
    rg_raw = statistical_region_growing(slice_img, seed=fg_seeds[0])
    rg_refined = refine_mask(rg_raw)
    dice_rg, iou_rg = compute_metrics(rg_refined, gt_mask)
    
    # 4. Run OpenCV Graph-Cut
    print("Running Graph-Cut (Energy Minimization)...")
    gc_raw = graph_cut_segmentation(slice_img, fg_seeds=fg_seeds, bg_seeds=bg_seeds)
    gc_refined = refine_mask(gc_raw)
    dice_gc, iou_gc = compute_metrics(gc_refined, gt_mask)
    
    print("-" * 50)
    print(f"Region Growing -> Dice: {dice_rg:.4f} | IoU: {iou_rg:.4f}")
    print(f"Graph-Cut      -> Dice: {dice_gc:.4f} | IoU: {iou_gc:.4f}")
    print("-" * 50)
    
    # 5. Plot Comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(slice_img, cmap="gray")
    axes[0].set_title("Original FLAIR")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    
    axes[2].imshow(rg_refined, cmap="gray")
    axes[2].set_title(f"Region Growing\nDice: {dice_rg:.2f}")
    
    axes[3].imshow(gc_refined, cmap="gray")
    axes[3].set_title(f"Graph-Cut (GrabCut)\nDice: {dice_gc:.2f}")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()