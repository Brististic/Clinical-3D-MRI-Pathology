import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nifti_volume(file_path):
    """Loads a 3D NIfTI volume and returns it as a NumPy array."""
    nii = nib.load(file_path)
    return nii.get_fdata(dtype=np.float32), nii.affine

def normalize_slice(slice_2d):
    """Min-Max normalization to standard [0, 255] grayscale range."""
    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)
    if max_val == min_val:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = (slice_2d - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

if __name__ == "__main__":
    # Base path matching your directory tree exactly
    base_path = "data/BraTS2021_00621/BraTS2021_00621"
    
    print("Loading NIfTI volumes...")
    flair_vol, _ = load_nifti_volume(f"{base_path}_flair.nii.gz")
    t1_vol, _    = load_nifti_volume(f"{base_path}_t1.nii.gz")
    t1ce_vol, _  = load_nifti_volume(f"{base_path}_t1ce.nii.gz")
    t2_vol, _    = load_nifti_volume(f"{base_path}_t2.nii.gz")
    seg_vol, _   = load_nifti_volume(f"{base_path}_seg.nii.gz")
    
    # Pick slice 80 (central slice where the tumor structure is visible)
    slice_idx = 80
    
    flair_slice = normalize_slice(flair_vol[:, :, slice_idx])
    t1_slice    = normalize_slice(t1_vol[:, :, slice_idx])
    t1ce_slice  = normalize_slice(t1ce_vol[:, :, slice_idx])
    t2_slice    = normalize_slice(t2_vol[:, :, slice_idx])
    seg_slice   = seg_vol[:, :, slice_idx]
    
    # Create the visualization window
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("FLAIR (Whole Tumor/Edema)")
    
    axes[1].imshow(t1_slice, cmap="gray")
    axes[1].set_title("T1")
    
    axes[2].imshow(t1ce_slice, cmap="gray")
    axes[2].set_title("T1-Gd (Enhancing Core)")
    
    axes[3].imshow(t2_slice, cmap="gray")
    axes[3].set_title("T2")
    
    # Overlay ground-truth tumor mask on top of FLAIR
    axes[4].imshow(flair_slice, cmap="gray")
    masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[4].imshow(masked_seg, cmap="autumn", alpha=0.7)
    axes[4].set_title("Ground Truth Mask Overlay")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()
    
    print(f"Success! Volume Dimensions: {flair_vol.shape}")