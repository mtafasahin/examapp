from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import base64
import io

app = FastAPI()

# CORS (gerekirse frontend için aç)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli yükle (model yolunu değiştir)
model = YOLO("runs/detect/train/weights/best.pt")

class ImageData(BaseModel):
    image_base64: str

@app.post("/predict")
def predict(data: ImageData):
    try:
        # Base64 başlığını ayıkla (data:image/...;base64, varsa)
        base64_data = data.image_base64
        if base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]
        # Base64'ten image oluştur
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Predict et
        results = model.predict(
            source=image,
            conf=0.09,
            save=False,
            save_txt=False,
            save_crop=False,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        width, height = result.orig_shape[1], result.orig_shape[0]

        predictions = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())

            predictions.append({
                "class_id": class_id,
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            })

        return {"success": True, "predictions": predictions}

    except Exception as e:
        return {"success": False, "error": str(e)}
