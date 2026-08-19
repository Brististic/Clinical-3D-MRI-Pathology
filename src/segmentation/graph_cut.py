import numpy as np
import cv2

def graph_cut_segmentation(img, fg_seeds=None, bg_seeds=None, rect=None):
    """
    Performs Graph-Cut (Energy Minimization) segmentation using OpenCV's GrabCut engine.
    Supports either seed points or a bounding region of interest (ROI).
    """
    # Convert single-channel to 3-channel BGR (required by GrabCut)
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()

    mask = np.zeros(img.shape[:2], dtype=np.uint8) # Default: GC_BGD (0)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if fg_seeds is not None and bg_seeds is not None:
        # Initialize with probable background/foreground
        mask[:] = cv2.GC_PR_BGD

        # Mark hard background seeds (0: cv2.GC_BGD)
        for r, c in bg_seeds:
            if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
                cv2.circle(mask, (c, r), radius=3, color=cv2.GC_BGD, thickness=-1)

        # Mark hard foreground/tumor seeds (1: cv2.GC_FGD)
        for r, c in fg_seeds:
            if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
                cv2.circle(mask, (c, r), radius=4, color=cv2.GC_FGD, thickness=-1)

        # Run Graph-Cut with mask initialization
        cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    elif rect is not None:
        # Run Graph-Cut with rectangle bounding box initialization
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Extract final foreground (both definite and probable foreground)
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return binary_mask