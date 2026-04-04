import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import our custom YOLO engine
from yolo_engine import YOLODetector

app = FastAPI(title="VerifAI Backend - YOLO Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def resolve_model_path() -> str:
    env_model_path = os.getenv("MODEL_PATH")
    candidates = [
        env_model_path,
        "weights/best.pt",
        "yolov5s.pt",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            return candidate

    # Keep a deterministic fallback and fail with a clear error in YOLODetector.
    return env_model_path or "weights/best.pt"


# Initialize the model once at startup
MODEL_PATH = resolve_model_path()
detector = YOLODetector(model_path=MODEL_PATH)

def calculate_credibility(detections: list) -> dict:
    """
    Calculates the credibility score based on detected AI artifacts.
    Base score is 100%. Deductions are made based on the number and confidence 
    of detected manipulations.
    """
    base_score = 100.0
    total_penalty = 0.0
    
    for det in detections:
        # Heavily penalize high-confidence anomalies
        penalty = det['confidence'] * 25.0 
        total_penalty += penalty
        
    final_score = max(0.0, base_score - total_penalty)
    
    # Determine classification based on score
    if final_score < 40.0:
        status = "Highly Likely AI/Manipulated"
    elif final_score < 80.0:
        status = "Suspicious"
    else:
        status = "Likely Real"
        
    return {
        "score": round(final_score, 2),
        "status": status,
        "anomalies_found": len(detections)
    }

@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Run YOLO Inference
        detections = detector.predict(image)

        # 2. Calculate Credibility Score
        credibility_report = calculate_credibility(detections)

        # 3. Format Response for the Frontend
        return {
            "filename": file.filename,
            "dimensions": image.size,
            "analysis": credibility_report,
            "detections": detections, # Array of bounding boxes to draw on the UI
            "model_path": MODEL_PATH,
            "message": "Analysis complete."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))