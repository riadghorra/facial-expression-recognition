import PIL as pl
import torch
import torchvision.transforms as transforms
import json
from classifier import Custom_vgg
import os
import pandas as pd


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


images = []
# r=root, d=directories, f = files
for r, d, f in os.walk(config["path_images"]):
    for file in f:
        if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
            images.append(os.path.join(r, file))

results_df = pd.DataFrame()

for path in images:
    print(path)
    image = pl.Image.open(path).resize(config["resolution"])
    x = pre_process(image).unsqueeze(0)
    model = Custom_vgg(1, config["cats"], DEVICE)
    # model.load_state_dict(torch.load(config["current_best_model"], map_location=DEVICE))
    # model.readable_output(x, config["catslist"])
    results = model.predict_single(x)
    results_dict = {cat: results[i] for i, cat in enumerate(config["catslist"])}
    results_dict["path"] = path
    results_df = results_df.append(results_dict, ignore_index=True)

results_df.to_csv("predictions.csv", index=False)

