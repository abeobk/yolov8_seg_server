"""
    TCP server to inference YOLO-SEG remotely
"""

print("Starting YOLO-SEG server..." );

import time
import json
import socket
import cv2
import numpy as np
import torch
from ultralytics import YOLO

HOST=""
PORT=44500
model_file = './model.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"

#HELPER
class TicToc:
    t0 = 0;
    #start timer
    def tic(self, msg = None):
        self.t0 = time.monotonic();
        if(msg is not None):
            print(msg + "...", end="")

    #stop timer
    def toc(self):
        dt = time.monotonic() - self.t0;
        print("("+str(round(dt,2))+" s)");

def send_ok(con:socket):
    con.sendall("OK\n".encode())

def send_err(con:socket, msg: str):
    con.sendall(f"ERROR: {msg}\n".encode())

def read_bytes(sock, n):
    buffer = bytearray(n)  # Initialize an empty buffer as a bytes object
    bytes_read = 0
    while bytes_read < n:
        # Read into the buffer starting at the current offset
        num_received = sock.recv_into(memoryview(buffer)[bytes_read:], min(512, n - bytes_read))
        print(f'received: {num_received}, total = {bytes_read}')
        if num_received == 0:
            raise RuntimeError("Socket connection closed before receiving all data")
        # Update the total number of bytes read so far
        bytes_read += num_received
    return buffer


tt = TicToc();



#LOAD MODELS
tt.tic("Loading segmentation model");
model = YOLO(model_file) # load pretrain model
tt.toc();

# the last image
img = None

#SERVER
print(f"Starting server...")
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    while True:
        try:
            print(f"Listerning on port {PORT}...")
            s.listen()
            con,client_addr = s.accept()
            with con:
                print(f"Connected by {client_addr}")
                connected=True
                while connected:
                    try:
                        jstr="";
                        #get a line
                        while connected:
                            data = con.recv(1)
                            c = chr(data[0])
                            #found new line, end of cmd
                            if(c=='\r' and chr(con.recv(1)[0])=='\n'):
                                break;
                            jstr+=c

                        if jstr:
                            print(">"+jstr)
                            cmd = json.loads(jstr)
                            if(cmd["name"]=="predict"):
                                imgsz = cmd["size"]
                                buf = bytearray(imgsz)
                                # send OK, start receiving
                                send_ok(con);
                                bytes_read=0
                                while bytes_read < imgsz:
                                    # Read into the buffer starting at the current offset
                                    num_received = con.recv_into(memoryview(buf)[bytes_read:], imgsz - bytes_read)
                                    if num_received == 0:
                                        raise RuntimeError("Socket connection closed before receiving all data")
                                    # Update the total number of bytes read so far
                                    bytes_read += num_received
                                img = cv2.imdecode(np.frombuffer(buf,dtype=np.uint8), cv2.IMREAD_COLOR)
                                if img is None:
                                    raise Exception("Invalid input image (img == None)")
                                min_score = cmd["min_score"]

                                send_ok(con)
                                # process image
                                tt.tic("Inferencing");
                                results = model(img, device = device);
                                tt.toc();
                                send_ok(con);
                                #collect all results
                                res = results[0]
                                # no result
                                if res.masks == None:
                                    raise Exception("No results"); 

                                names = res.names
                                boxes = res.boxes.cpu()
                                labels = boxes.cls
                                # result json
                                jres={}
                                jres["Names"] = res.names # classes
                                jres["InputShape"]=res.masks.orig_shape
                                jres["OutputShape"]=(res.masks.shape[1],res.masks.shape[2])
                                cls = res.boxes.cpu().cls.numpy() #result count
                                confs = res.boxes.cpu().conf.numpy()
                                jmask = jres["Masks"] = [] 
                                res_id = 0
                                for i,mask in enumerate(res.masks):
                                    if confs[i] < min_score:
                                        break;

                                    mask = np.squeeze(mask.data.cpu())
                                    contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    contour = contours[0]
                                    peri = cv2.arcLength(contour, True)
                                    area = cv2.contourArea(contour)
                                    contour = np.squeeze(cv2.approxPolyDP(contour, 0.01 * peri, True)).tolist()

                                    if len(contour) < 3:
                                        continue

                                    jmask.append({})
                                    ji = jmask[res_id]
                                    ji["Class"] = int(float(cls[i]))
                                    ji["Length"] = peri
                                    ji["Area"] = area
                                    ji["Confident"] = round(float(confs[i]),2)
                                    ji["Contour"]=contour
                                    res_id+=1
                                    
                                con.sendall((json.dumps(jres,indent=None)+'\n').encode())
                                send_ok(con)
                            else:
                                print(f"Unknown command {jstr}")
                    except Exception as ex:
                        print("ERROR: " + str(ex))
                        send_err(con,str(ex))
                    time.sleep(0.010); #100 hz polling
        except Exception as ex: 
            print("Opps! "+str(ex))