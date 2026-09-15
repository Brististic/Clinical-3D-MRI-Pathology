from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from src.io.data_loader import (
    get_normalized_slice,
    load_nifti_bytes,
    validate_volume_pair,
)
from src.metrics.evaluation import compute_metrics
from src.segmentation.graph_cut import graph_cut_segmentation
from src.segmentation.region_growing import refine_mask, statistical_region_growing


def test_get_normalized_slice_scales_to_uint8():
    volume = np.zeros((10, 10, 3), dtype=np.float32)
    volume[:, :, 1] = np.linspace(10.0, 30.0, 100, dtype=np.float32).reshape(10, 10)

    slice_img = get_normalized_slice(volume, 1)

    assert slice_img.shape == (10, 10)
    assert slice_img.dtype == np.uint8
    assert slice_img.min() == 0
    assert slice_img.max() == 255


def test_statistical_region_growing_marks_seed_region():
    img = np.zeros((40, 40), dtype=np.float32)
    img[12:28, 12:28] = 120.0

    mask = statistical_region_growing(img, seed=(20, 20), std_multiplier=2.0)

    assert mask.shape == img.shape
    assert mask[20, 20] == 1
    assert mask.sum() > 20


def test_refine_mask_keeps_largest_connected_component():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[3:12, 3:12] = 1
    mask[18:25, 18:25] = 1
    mask[20:22, 20:22] = 0

    refined = refine_mask(mask)

    assert refined.shape == mask.shape
    assert refined.dtype == np.uint8
    assert refined.sum() > 0
    assert refined.sum() < mask.sum()


def test_graph_cut_segmentation_returns_binary_mask():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[18:46, 18:46] = 180

    fg_seeds = [(30, 30), (35, 35)]
    bg_seeds = [(10, 10), (10, 55), (55, 10), (55, 55)]

    mask = graph_cut_segmentation(img, fg_seeds=fg_seeds, bg_seeds=bg_seeds)

    assert mask.shape == img.shape
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0


def test_load_nifti_bytes_reads_uploaded_volume():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "volume.nii"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        file_data = path.read_bytes()

    loaded, header = load_nifti_bytes(file_data)

    np.testing.assert_array_equal(loaded, data)
    assert header.get_zooms()[:3] == (1.0, 1.0, 1.0)


def test_validate_volume_pair_rejects_mismatched_shapes():
    with np.testing.assert_raises(ValueError):
        validate_volume_pair(np.zeros((4, 4, 4)), np.zeros((4, 4, 3)))


def test_compute_metrics_rejects_mismatched_masks():
    with np.testing.assert_raises(ValueError):
        compute_metrics(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8))
