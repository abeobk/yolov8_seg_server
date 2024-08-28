
import numpy as np
from ultralytics import YOLO


model = YOLO('model.pt') # load pretrain model
model.export(format='onnx')

