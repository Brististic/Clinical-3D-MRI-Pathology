import nibabel as nib
import numpy as np

def load_volume(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(dtype=np.float32), nii.header

def get_normalized_slice(volume, slice_idx):
    slice_data = volume[:, :, slice_idx]
    min_v, max_v = np.min(slice_data), np.max(slice_data)
    if max_v > min_v:
        return ((slice_data - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    return np.zeros_like(slice_data, dtype=np.uint8)