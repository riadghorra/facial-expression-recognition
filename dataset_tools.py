import PIL as pl
import numpy as np
import torch
import torchvision.transforms as transforms

size = 48
emotions = 7

def pixelstring_to_numpy(string, flatten = False):
    pixels = string.split()
    if flatten :
        out = np.array([int(i) for i in pixels])
        return out
    out = np.zeros((size,size))
    for i in range(size):
        out[i] = np.array([int(k) for k in pixels[size*i:size*(i+1)]])
    return out

def pixelstring_to_torchtensor(string, datatype = torch.uint8, flatten = False ):
    return torch.tensor(pixelstring_to_numpy(string, flatten = flatten), dtype = datatype)
    
    
def string_to_pilimage(pixelstring):
    imarray = pixelstring_to_numpy(pixelstring)
    out = pl.Image.fromarray(imarray).convert("L")
    return out

def tensor_to_pilimage(tensor, resolution = (256,256)):
    im = transforms.ToPILImage()(tensor.unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im

def create_datatensor(dataset, sample = None):
    out = torch.tensor([], dtype = torch.uint8)
    for tensor in dataset["tensors"][:sample]:
        out = torch.cat((out, tensor.unsqueeze(0)))
    return out

def label_to_vector(label):
    out = torch.zeros(emotions)
    out[label] = 1
    return out

