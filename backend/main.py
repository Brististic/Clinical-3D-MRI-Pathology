from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from services import MedicalImagingService

app = FastAPI(title="Medical Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATIENT_ID = "BraTS2021_00621"
BASE_PATH = os.path.abspath(f"data/{PATIENT_ID}/{PATIENT_ID}")

# Initialize service
service = MedicalImagingService(
    flair_path=f"{BASE_PATH}_flair.nii.gz",
    seg_path=f"{BASE_PATH}_seg.nii.gz"
)

class SegmentRequest(BaseModel):
    slice_idx: int
    seed_r: int
    seed_c: int

@app.get("/api/metadata")
def get_metadata():
    return {
        "patient_id": PATIENT_ID,
        "total_slices": service.num_slices,
        "voxel_dimensions_mm": list(service.voxel_dims),
        "unit_voxel_vol_mm3": service.voxel_vol_mm3
    }

@app.get("/api/slice/{slice_idx}")
def get_slice(slice_idx: int, mode: str = "grayscale"):
    img, gt = service.get_slice_data(slice_idx)
    img_b64 = service.matrix_to_base64(img, colormap=mode)
    gt_b64 = service.matrix_to_base64((gt * 255).astype("uint8")) if gt is not None else None
    
    return {
        "slice_idx": slice_idx,
        "image": f"data:image/png;base64,{img_b64}",
        "ground_truth": f"data:image/png;base64,{gt_b64}" if gt_b64 else None
    }

@app.post("/api/segment")
def run_segmentation(req: SegmentRequest):
    mask, dice, iou = service.segment_slice_region_growing(req.slice_idx, req.seed_r, req.seed_c)
    mask_b64 = service.matrix_to_base64((mask * 255).astype("uint8"))
    
    return {
        "mask": f"data:image/png;base64,{mask_b64}",
        "dice": round(dice, 4),
        "iou": round(iou, 4)
    }

@app.get("/api/volume-metrics")
def get_volume_metrics():
    pred_vol, gt_vol = service.compute_full_volume()
    return {
        "predicted_volume_cm3": round(pred_vol, 2),
        "ground_truth_volume_cm3": round(gt_vol, 2),
        "error_cm3": round(abs(pred_vol - gt_vol), 2)
    }