from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(data="lvis.yaml", epochs=1, imgsz=640)
