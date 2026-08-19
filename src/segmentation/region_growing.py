import numpy as np
import cv2
from collections import deque

def statistical_region_growing(img, seed, std_multiplier=1.7, max_iter=25000):
    rows, cols = img.shape
    segmented_mask = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)
    
    queue = deque([seed])
    visited[seed[0], seed[1]] = True
    region_pixels = [float(img[seed[0], seed[1]])]
    
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
    
    count = 0
    while queue and count < max_iter:
        r, c = queue.popleft()
        segmented_mask[r, c] = 1
        count += 1
        
        mean_val = np.mean(region_pixels)
        std_val = max(np.std(region_pixels), 8.0)
        
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                pixel_val = float(img[nr, nc])
                if abs(pixel_val - mean_val) <= (std_multiplier * std_val) and pixel_val > 20:
                    region_pixels.append(pixel_val)
                    queue.append((nr, nc))
                    
    return segmented_mask

def refine_mask(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        opened = (labels == largest_label).astype(np.uint8)
    return opened