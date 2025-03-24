python scripts/predict_to_json.py

yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=4 device=cpu

# bir önceki train'in bilgilerini kullanarak devam etme 
yolo detect train data=data/dataset.yaml model=runs/detect/train/weights/best.pt epochs=30 imgsz=640 batch=2 device=cpu

# test
yolo detect predict model=runs/detect/train/weights/best.pt source=data/images/image.png conf=0.25

apt update && apt install tree -y
tree -L 2