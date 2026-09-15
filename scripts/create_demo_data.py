"""Create a small synthetic NIfTI study for local UI and pipeline testing.

This data is synthetic and must not be used for clinical validation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


PATIENT_ID = "BraTS2021_00621"
VOLUME_SHAPE = (128, 128, 32)


def create_demo_volume() -> tuple[np.ndarray, np.ndarray]:
    rows, cols, slices = np.indices(VOLUME_SHAPE, dtype=np.float32)
    center_r, center_c = (VOLUME_SHAPE[0] - 1) / 2, (VOLUME_SHAPE[1] - 1) / 2

    brain = (
        ((rows - center_r) / 48.0) ** 2
        + ((cols - center_c) / 52.0) ** 2
        <= 1.0
    )
    slice_profile = 0.65 + 0.35 * np.cos(
        ((slices - (VOLUME_SHAPE[2] - 1) / 2) / 16.0) * np.pi / 2
    )
    rng = np.random.default_rng(42)
    flair = (brain * (75.0 * slice_profile + rng.normal(0, 4, VOLUME_SHAPE))).astype(
        np.float32
    )

    lesion = (
        (((rows - 70.0) / 13.0) ** 2)
        + (((cols - 82.0) / 17.0) ** 2)
        + (((slices - 16.0) / 7.0) ** 2)
        <= 1.0
    ) & brain
    flair[lesion] += 115.0

    return flair, lesion.astype(np.uint8)


def main() -> None:
    output_dir = Path("data") / PATIENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    flair, segmentation = create_demo_volume()
    affine = np.diag([1.0, 1.0, 3.0, 1.0])

    nib.save(
        nib.Nifti1Image(flair, affine),
        output_dir / f"{PATIENT_ID}_flair.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        output_dir / f"{PATIENT_ID}_seg.nii.gz",
    )

    print(f"Created synthetic demo study in {output_dir}")
    print(f"Volume shape: {flair.shape}")
    print(f"Synthetic lesion voxels: {int(segmentation.sum()):,}")
    print("This data is for UI and pipeline testing only, not clinical use.")


if __name__ == "__main__":
    main()
