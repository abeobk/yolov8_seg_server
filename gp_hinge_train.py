from ultralytics import YOLO

#train GP hinge bin picking
# model = YOLO("yolov8m-seg.pt") # load pretrain model for training
model = YOLO("best_gp_hinge.pt") # load pretrain model for training
results = model.train(data="./gp_hinge_dataset.yaml",epochs=300,imgsz=640,device='cuda',
                      augment=True, scale=0.1, translate = 0.25, degrees=10)
    