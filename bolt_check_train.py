from ultralytics import YOLO

#train GP hinge bin picking
# model = YOLO("yolov8m-seg.pt") # load pretrain model for training
# model = YOLO("yolov8s.pt") # load pretrain model for training
model = YOLO("best_bolt_check_s.pt") # load pretrain model for training
results = model.train(data="./bolt_check_dataset.yaml",epochs=300,imgsz=640,device='cuda',
                      augment=True, scale=0.05, translate = 0.1, degrees=3, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=0.0,flipud=0.0, fliplr=0.0)
    