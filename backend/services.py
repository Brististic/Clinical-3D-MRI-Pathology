import base64
import io
import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from collections import deque

class MedicalImagingService:
    def __init__(self, flair_path, seg_path=None):
        self.flair_vol = nib.load(flair_path).get_fdata(dtype=np.float32)
        self.header = nib.load(flair_path).header
        self.voxel_dims = self.header.get_zooms()[:3]
        self.voxel_vol_mm3 = float(self.voxel_dims[0] * self.voxel_dims[1] * self.voxel_dims[2])
        
        self.gt_vol = None
        if seg_path:
            self.gt_vol = (nib.load(seg_path).get_fdata(dtype=np.float32) > 0).astype(np.uint8)
            
        self.num_slices = self.flair_vol.shape[2]

    def get_slice_data(self, slice_idx):
        slice_idx = max(0, min(slice_idx, self.num_slices - 1))
        raw_slice = self.flair_vol[:, :, slice_idx]
        
        min_v, max_v = float(np.min(raw_slice)), float(np.max(raw_slice))
        if max_v > min_v:
            norm_slice = ((raw_slice - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            norm_slice = np.zeros_like(raw_slice, dtype=np.uint8)
            
        gt_slice = self.gt_vol[:, :, slice_idx] if self.gt_vol is not None else None
        return norm_slice, gt_slice

    def segment_slice_region_growing(self, slice_idx, seed_r, seed_c, std_multiplier=1.7):
        img, gt = self.get_slice_data(slice_idx)
        rows, cols = img.shape
        mask = np.zeros((rows, cols), dtype=np.uint8)
        visited = np.zeros((rows, cols), dtype=bool)

        if not (0 <= seed_r < rows and 0 <= seed_c < cols):
            return mask, 0.0, 0.0

        queue = deque([(seed_r, seed_c)])
        visited[seed_r, seed_c] = True
        region_pixels = [float(img[seed_r, seed_c])]

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     ( 0, -1),          ( 0, 1),
                     ( 1, -1), ( 1, 0), ( 1, 1)]

        count = 0
        while queue and count < 25000:
            r, c = queue.popleft()
            mask[r, c] = 1
            count += 1

            mean_val = np.mean(region_pixels)
            std_val = max(np.std(region_pixels), 8.0)

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    visited[nr, nc] = True
                    val = float(img[nr, nc])
                    if abs(val - mean_val) <= (std_multiplier * std_val) and val > 20:
                        region_pixels.append(val)
                        queue.append((nr, nc))

        # Morphological post-processing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        refined_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        # Metrics
        dice, iou = 0.0, 0.0
        if gt is not None:
            intersection = np.sum((refined_mask == 1) & (gt == 1))
            tot_pred = np.sum(refined_mask == 1)
            tot_gt = np.sum(gt == 1)
            if tot_pred + tot_gt > 0:
                dice = float((2.0 * intersection) / (tot_pred + tot_gt))
                union = tot_pred + tot_gt - intersection
                iou = float(intersection / union) if union > 0 else 0.0

        return refined_mask, dice, iou

    def compute_full_volume(self):
        pred_voxel_count = 0
        gt_voxel_count = int(np.sum(self.gt_vol == 1)) if self.gt_vol is not None else 0

        for z in range(self.num_slices):
            slice_gt = self.gt_vol[:, :, z] if self.gt_vol is not None else None
            if slice_gt is not None and np.sum(slice_gt) > 15:
                gt_coords = np.argwhere(slice_gt == 1)
                img, _ = self.get_slice_data(z)
                vals = [img[r, c] for r, c in gt_coords]
                best_seed = tuple(gt_coords[np.argmax(vals)])
                mask, _, _ = self.segment_slice_region_growing(z, best_seed[0], best_seed[1])
                pred_voxel_count += int(np.sum(mask == 1))

        pred_vol_cm3 = (pred_voxel_count * self.voxel_vol_mm3) / 1000.0
        gt_vol_cm3 = (gt_voxel_count * self.voxel_vol_mm3) / 1000.0
        return pred_vol_cm3, gt_vol_cm3

    @staticmethod
    def matrix_to_base64(img_array, colormap=None):
        if colormap == "heatmap":
            colored = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
            img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")