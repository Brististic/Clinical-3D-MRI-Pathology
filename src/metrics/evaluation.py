import numpy as np


def compute_metrics(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Prediction and reference masks must have matching dimensions.")

    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    total_pred = np.sum(pred_mask == 1)
    total_gt = np.sum(gt_mask == 1)
    if total_pred + total_gt == 0:
        return 1.0, 1.0

    dice = (2.0 * intersection) / (total_pred + total_gt)
    union = total_pred + total_gt - intersection
    iou = intersection / union if union > 0 else 0.0
    return float(dice), float(iou)
