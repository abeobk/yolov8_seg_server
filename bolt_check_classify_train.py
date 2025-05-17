from ultralytics import YOLO

#train GP hinge bin picking
model = YOLO("yolov8s-cls.pt") # load pretrain model for training
results = model.train(data="./bolt_check_classify_dataset.yaml", epochs=300, imgsz=640)
    