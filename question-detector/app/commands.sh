python scripts/predict_to_json.py

yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=4 device=cpu

# bir önceki train'in bilgilerini kullanarak devam etme 
yolo detect train data=data/dataset.yaml model=runs/detect/train/weights/best.pt epochs=30 imgsz=640 batch=2 device=cpu

# test
yolo detect predict model=runs/detect/train/weights/best.pt source=data/images/image.png conf=0.25

apt update && apt install tree -y
tree -L 2

source venv/bin/activate

uvicorn main:app --host 0.0.0.0 --port 8080


yolo detect train \
  model=runs/detect/train-only-q/weights/best.pt \
  data=data/dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=1 \
  project=runs/detect \
  name=train-only-q-v2 \
  device=cpu

yolo detect train \
  model=yolov8n.pt \
  data=data/dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=1 \
  name=train-only-q \
  device=cpu

source venv/bin/activate

yolo detect train \
  model=runs/detect/train-only-q-v3/weights/best.pt \
  data=data/dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=1 \
  project=runs/detect \
  name=train-only-q-v4 \
  device=cpu


for img in *.jpg; do   txt_file="../labels/${img%.jpg}.txt";   if [ ! -f "$txt_file" ]; then     echo "Removing $img (no label)";     rm "$img";   fi; done
for img in *.png; do   txt_file="../labels/${img%.jpg}.txt";   if [ ! -f "$txt_file" ]; then     echo "Removing $img (no label)";     rm "$img";   fi; done

yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=1 name=train-answers device=cpu

apt-get update
apt-get install libzbar0 

