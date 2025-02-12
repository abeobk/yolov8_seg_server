from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("best.pt") # load pretrain model for training
#model = YOLO("yolov8l-seg.pt") # load pretrain model for training
results = model.train(data="./dataset.yaml",epochs=300,imgsz=640,device='cuda',
                      augment=True)
