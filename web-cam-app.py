from pipeline import process_frame
from classifier import Custom_vgg
import torch
import json
import cv2
import PIL as pl

with open('config.json') as json_file:
    config = json.load(json_file)



cap = cv2.VideoCapture(0)
model = Custom_vgg(1, len(config["catslist"]), torch.device("cpu"))
model.load_state_dict(torch.load(config["current_best_model"], map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True: 
            out = process_frame(model, frame)
            cv2.imshow('title', out)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()