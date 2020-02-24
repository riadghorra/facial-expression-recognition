import PIL as pl
import torch
import torchvision.transforms as transforms
import json
from classifier import Custom_vgg
import os
import pandas as pd
import numpy as np

from pipeline import crop_faces, crop_cv_img

with open('config.json') as json_file:
    config = json.load(json_file)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Initialisation de cuda')
    torch.cuda.init()
else:
    print('Mode CPU')
    DEVICE = torch.device('cpu')

pre_process = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])


def resize_input_image(pil_img):
    (w, h) = pil_img.size
    aspect_ratio = w / h

    if w > 800:
        return pil_img.resize((800, int(800 / aspect_ratio)))

    if h > 800:
        return pil_img.resize((int(800 * aspect_ratio), 800))

    return pil_img


images = []
# r=root, d=directories, f = files
for r, d, f in os.walk(config["path_images"]):
    for file in f:
        if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
            images.append(os.path.join(r, file))

results_df = pd.DataFrame()

IMG_TO_TEST = 200
i = 0

for path in images:
    if i >= IMG_TO_TEST:
        break
    i += 1
    print("Processing {}".format(path))
    image = pl.Image.open(path)
    image = resize_input_image(image)
    cv_img = np.array(image)
    [face_coords] = crop_faces([cv_img])
    if face_coords is not None:
        print("Face found on image.")
        (x, y, w, h) = face_coords
        face_image = crop_cv_img(cv_img, x, y, w, h)
        pil_face_image = pl.Image.fromarray(face_image).resize(config["resolution"])
        pil_face_image = pre_process(pil_face_image).unsqueeze(0)

        model = Custom_vgg(1, config["cats"], DEVICE)
        model.load_state_dict(torch.load(config["current_best_model"], map_location=DEVICE))
        # model.readable_output(x, config["catslist"])
        results = model.predict_single(pil_face_image)
        results_dict = {cat: results[i] for i, cat in enumerate(config["catslist"])}
        results_dict["path"] = path
        results_df = results_df.append(results_dict, ignore_index=True)

results_df.to_csv("predictions.csv", index=False)

