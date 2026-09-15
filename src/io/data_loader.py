import io

import nibabel as nib
import numpy as np


def load_volume(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(dtype=np.float32), nii.header


def load_nifti_bytes(file_data):
    if hasattr(file_data, "getvalue"):
        file_data = file_data.getvalue()
    image = nib.Nifti1Image.from_file_map(
        {
            "header": nib.FileHolder(),
            "image": nib.FileHolder(fileobj=io.BytesIO(file_data)),
        }
    )
    return image.get_fdata(dtype=np.float32), image.header


def validate_volume_pair(image_volume, segmentation_volume):
    if image_volume.ndim != 3 or segmentation_volume.ndim != 3:
        raise ValueError("FLAIR and segmentation files must both contain 3D volumes.")
    if image_volume.shape != segmentation_volume.shape:
        raise ValueError(
            "FLAIR and segmentation volumes must have matching dimensions "
            f"(got {image_volume.shape} and {segmentation_volume.shape})."
        )


def get_normalized_slice(volume, slice_idx):
    slice_data = volume[:, :, slice_idx]
    min_v, max_v = np.min(slice_data), np.max(slice_data)
    if max_v > min_v:
        return ((slice_data - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    return np.zeros_like(slice_data, dtype=np.uint8)