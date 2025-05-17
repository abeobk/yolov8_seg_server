from ultralytics import YOLO
import numpy as np
import cv2

# model = YOLO("yolov8l-seg.pt") # load pretrain model for training
model = YOLO("yolov8m-obb.pt") # load pretrain model for training
results = model.train(data="./bolt1_obb.yaml",epochs=300,imgsz=640,device='cuda',
                      augment=True, scale=0.2, translate = 0.5, degrees=180, flipud=0.5, fliplr=0.5)
