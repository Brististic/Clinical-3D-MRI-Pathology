import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from src.io.data_loader import load_volume, get_normalized_slice
from src.segmentation.region_growing import statistical_region_growing, refine_mask

# Load data
patient_id = "BraTS2021_00621"
base_path = f"data/{patient_id}/{patient_id}"

flair_vol, _ = load_volume(f"{base_path}_flair.nii.gz")
seg_vol, _   = load_volume(f"{base_path}_seg.nii.gz")
gt_vol       = (seg_vol > 0).astype(np.uint8)

num_slices = flair_vol.shape[2]
initial_slice = 80

# Setup Figure
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
plt.subplots_adjust(bottom=0.2)

# Initial Render
flair_slice = get_normalized_slice(flair_vol, initial_slice)
gt_slice    = gt_vol[:, :, initial_slice]

im0 = axes[0].imshow(flair_slice, cmap="gray")
axes[0].set_title("Original FLAIR")
axes[0].axis("off")

im1 = axes[1].imshow(gt_slice, cmap="autumn")
axes[1].set_title("Ground Truth Mask")
axes[1].axis("off")

# Overlay
im2 = axes[2].imshow(flair_slice, cmap="gray")
masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
im2_overlay = axes[2].imshow(masked_gt, cmap="autumn", alpha=0.6)
axes[2].set_title(f"Axial Slice: {initial_slice}")
axes[2].axis("off")

# Add Slider
ax_slider = plt.axes([0.25, 0.08, 0.50, 0.04])
slider = Slider(ax_slider, "Slice", 0, num_slices - 1, valinit=initial_slice, valstep=1)

def update(val):
    idx = int(slider.val)
    f_slice = get_normalized_slice(flair_vol, idx)
    g_slice = gt_vol[:, :, idx]
    
    im0.set_data(f_slice)
    im1.set_data(g_slice)
    im2.set_data(f_slice)
    
    masked = np.ma.masked_where(g_slice == 0, g_slice)
    im2_overlay.set_data(masked)
    
    axes[2].set_title(f"Axial Slice: {idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()