from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import base64
import io
from io import BytesIO
import uuid
import json
import os
import logging

app = FastAPI()


# Basit log yapılandırması (konsola yazdırır)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

IMAGES_DIR = "data/images"
ANSWERS_DIR = "data/answers"
QUESTIONS_JSON_PATH = "data/json/questions.json"
ANSWERS_JSON_PATH = "data/json/answers.json"
IMAGE_URL_PREFIX = "../data/images"
CROPS_DIR = Path("data/crops")

# Ensure dirs exist
os.makedirs(IMAGES_DIR, exist_ok=True)
Path(QUESTIONS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
if not os.path.exists(QUESTIONS_JSON_PATH):
    with open(QUESTIONS_JSON_PATH, "w") as f:
        json.dump([], f)

# Request body model
class ImageData(BaseModel):
    image_base64: str

class QuestionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class UploadQuestionsRequest(BaseModel):
    imageData: ImageData
    questions: List[QuestionBox]


# CORS (gerekirse frontend için aç)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli yükle (model yolunu değiştir)
model = YOLO("runs/detect/train-only-q-v4/weights/best.pt")

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
        logger.info(f"Received image with shape: {image.size}")
        # Predict et
        results = model.predict(
            source=image,
            conf=0.25,
            save=False,
            save_txt=False,
            save_crop=False,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        width, height = result.orig_shape[1], result.orig_shape[0]
        logger.info(f"Detected {len(boxes)} objects")
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


@app.post("/send-to-fix")
def upload_questions(payload: UploadQuestionsRequest):
    try:
        # 👇 Base64 temizleme
        base64_str = payload.imageData.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Görseli decode et ve kaydet
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(IMAGES_DIR, filename)
        image.save(image_path)

        image_url = f"{IMAGE_URL_PREFIX}/{filename}"
        logger.info(f"Image Added : {image_url}")
        crops = []
        # Yeni question objelerini oluştur
        new_entries = []
        for idx, q in enumerate(payload.questions):
            entry = {
                "question": {
                    "x": q.x,
                    "y": q.y,
                    "width": q.width,
                    "height": q.height,
                    "imageUrl": image_url
                }
            }

            left = q.x
            upper = q.y
            right = q.x + q.width
            lower = q.y + q.height

            crop = image.crop((left, upper, right, lower))

            crop_filename = f"{filename}_q{idx + 1}.jpg"
            crop_path = CROPS_DIR / crop_filename
            crop.save(crop_path)

            crops.append(str(crop_path))

            new_entries.append(entry)
            logger.info(f"Added question: {entry['question']}")

        # Mevcut questions.json dosyasına ekle
        with open(QUESTIONS_JSON_PATH, "r+", encoding="utf-8") as f:
            current_data = json.load(f)
            current_data.extend(new_entries)
            f.seek(0)
            json.dump(current_data, f, indent=2, ensure_ascii=False)
            f.truncate()

        logger.info(f"Successfully appended {len(new_entries)} questions to {QUESTIONS_JSON_PATH}")


        return {
            "success": True,
            "added": len(new_entries),
            "imageFile": filename,
            "crops": crops
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/send-to-fix-for-answers")
def upload_answers(payload: UploadQuestionsRequest):
    try:
        # 👇 Base64 temizleme
        base64_str = payload.imageData.image_base64
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Görseli decode et ve kaydet
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(ANSWERS_DIR, filename)
        image.save(image_path)

        image_url = f"{IMAGE_URL_PREFIX}/{filename}"
        logger.info(f"Image Added : {image_url}")
        # Yeni question objelerini oluştur
        new_entries = []
        for idx, q in enumerate(payload.questions):
            entry = {
                "question": {
                    "x": q.x,
                    "y": q.y,
                    "width": q.width,
                    "height": q.height,
                    "imageUrl": image_url
                }
            }

            new_entries.append(entry)
            logger.info(f"Added question: {entry['question']}")

        # Mevcut questions.json dosyasına ekle
        with open(ANSWERS_JSON_PATH, "r+", encoding="utf-8") as f:
            current_data = json.load(f)
            current_data.extend(new_entries)
            f.seek(0)
            json.dump(current_data, f, indent=2, ensure_ascii=False)
            f.truncate()

        logger.info(f"Successfully appended {len(new_entries)} answers to {ANSWERS_JSON_PATH}")


        return {
            "success": True,
            "added": len(new_entries),
            "imageFile": filename
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))    