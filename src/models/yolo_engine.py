from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLO model on {self.device}...")

        try:
            weights_path = Path(model_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")

            # Load your trained detector weights (YOLOv8/YOLOv5 .pt exported by training).
            self.model = YOLO(str(weights_path))
            print(f"Model loaded successfully from {weights_path}.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def predict(self, image: Image.Image, confidence_threshold=0.45):
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Run inference and keep only detections over the configured confidence threshold.
        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            device=0 if self.device.type == "cuda" else "cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            names = result.names or {}
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item()) if box.cls is not None else -1
                class_name = names.get(class_id, str(class_id))

                detections.append({
                    "class_name": class_name,
                    "confidence": float(box.conf[0].item()),
                    "box": {
                        "xmin": float(xyxy[0]),
                        "ymin": float(xyxy[1]),
                        "xmax": float(xyxy[2]),
                        "ymax": float(xyxy[3]),
                    },
                })

        return detections