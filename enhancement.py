import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib

def load_slice(file_path, slice_idx=80):
    """Loads a specific 2D axial slice from a NIfTI volume."""
    vol = nib.load(file_path).get_fdata(dtype=np.float32)
    slice_data = vol[:, :, slice_idx]
    
    # Normalize to [0, 255] uint8
    min_val, max_val = np.min(slice_data), np.max(slice_data)
    if max_val == min_val:
        return np.zeros_like(slice_data, dtype=np.uint8)
    norm = (slice_data - min_val) / (max_val - min_val) * 255.0
    return norm.astype(np.uint8)

# --- 1. Spatial Domain: CLAHE ---
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

# --- 2. Frequency Domain: 2D-DFT & Gaussian High-Pass Filter ---
def apply_frequency_highpass(img, cutoff_d0=30):
    """
    Transforms the image to the frequency domain using 2D-DFT,
    applies a Gaussian High-Pass Filter (GHPF) to emphasize edges/textures,
    and returns the reconstructed spatial image along with the magnitude spectrum.
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Compute 2D Discrete Fourier Transform & shift zero frequency to center
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate Magnitude Spectrum for visualization: 20 * log(|F(u,v)| + 1)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Create Gaussian High-Pass Filter mask H(u, v) = 1 - exp(-D^2 / (2 * D0^2))
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    dist_matrix = np.sqrt((u - crow)**2 + (v - ccol)**2)
    
    ghpf_mask = 1.0 - np.exp(-(dist_matrix**2) / (2 * (cutoff_d0**2)))
    
    # Apply filter in frequency domain
    filtered_shift = dft_shift * ghpf_mask
    
    # Inverse Shift and Inverse 2D-DFT to return to spatial domain
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize result to [0, 255]
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back, magnitude_spectrum

if __name__ == "__main__":
    patient_id = "BraTS2021_00621"
    flair_path = f"data/{patient_id}/{patient_id}_flair.nii.gz"
    
    # 1. Load original slice
    original_slice = load_slice(flair_path, slice_idx=80)
    
    # 2. Spatial Enhancement
    enhanced_clahe = apply_clahe(original_slice, clip_limit=3.0)
    
    # 3. Frequency Domain Filtering
    edge_highpass, spectrum = apply_frequency_highpass(original_slice, cutoff_d0=25)
    
    # 4. Display Comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title("1. Original FLAIR Slice")
    
    axes[1].imshow(enhanced_clahe, cmap="gray")
    axes[1].set_title("2. Spatial Domain: CLAHE")
    
    axes[2].imshow(spectrum, cmap="inferno")
    axes[2].set_title("3. 2D-DFT Magnitude Spectrum")
    
    axes[3].imshow(edge_highpass, cmap="gray")
    axes[3].set_title("4. Frequency Domain: GHPF Filtered")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()