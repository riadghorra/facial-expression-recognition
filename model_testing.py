import pil as pl
import torch
import torchvision.transforms as transforms
import json
from Classifier import Custom_vgg


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
x = pre_process(pl.Image.open(config["path_images"]).resize(config["resolution"]))


model = Custom_vgg(1, config["cats"], DEVICE)

print(model.readable_output(x, config["catslist"]))