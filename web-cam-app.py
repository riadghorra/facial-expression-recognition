from pipeline import process_frame, config
from classifier import Custom_vgg
import torch
import cv2


cap = cv2.VideoCapture(0)
model = Custom_vgg(1, len(config["catslist"]), torch.device("cpu"))
model.load_state_dict(torch.load(config["current_best_model"], map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if ret:
            out = process_frame(model, frame)
            cv2.imshow('The facial expression challenge!', out)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()