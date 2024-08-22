from ultralytics import YOLO
import numpy as np
import cv2
import glob
import json
from scipy.interpolate import splprep, splev

files = []
for file in glob.glob('./datasets/bolt_pick/images/val/*.PNG'):
    print(file)
    files.append(file)

model = YOLO("bolt_pick_yolov8l_seg.pt") # load pretrain model
# model = YOLO("bolt_pick_yolov8l_seg.pt") # load pretrain model
model.export(format="onnx")

random_colors = [(int(np.random.randint(0, 256)), 
                  int(np.random.randint(0, 256)), 
                  int(np.random.randint(0, 256))) for _ in range(1024)]

for file in files:
    frame = cv2.imread(file)

    results = model(frame,device='cpu');
    res = results[0]
    if res.masks == None:
        continue

    names = res.names
    boxes = res.boxes.cpu()
    labels = boxes.cls
    j={}
    for i,mask in enumerate(res.masks):
        color =  random_colors[i]
        mask = mask.data.cpu().numpy().astype(np.uint8) # Ensure mask is binary
        mask = mask.reshape((mask.shape[1],mask.shape[2]))
        mask = cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.squeeze(contours)
        j[i] = contours
        print(j[i])
        label = names[int(labels[i].item())]
        x0,y0,x1,y1 = boxes.xyxy[i]
        conf = boxes.conf[i]
        cv2.polylines(frame,contours,True,color,thickness=2)
        cv2.putText(frame,label+', '+str(conf),(int((x0.item()+x1.item())/2),int((y0.item()+y1.item())/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.imshow("img",frame)
    if cv2.waitKey() == 'q':
        break
