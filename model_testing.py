import PIL as pl
import torch
import torchvision.transforms as transforms
import json
from classifier import Custom_vgg


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
image = pl.Image.open(config["path_images"]).resize(config["resolution"])
x = pre_process(image).unsqueeze(0)


model = Custom_vgg(1, config["cats"], DEVICE)
model.load_state_dict(torch.load("current_best_model", map_location=DEVICE))

print(model.readable_output(x, config["catslist"]))
