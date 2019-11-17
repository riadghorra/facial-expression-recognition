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


def pixelstring_to_tensor_vgg16(pixels):
    # resize image to match input layer of vgg16
    img = string_to_pilimage(pixels).resize((224, 224))
    # input needs to have 3 channels
    tensor = transforms.ToTensor()(img).repeat((3, 1, 1))

    # TODO: rescale image pixels to range of [0, 1] then normalise with
    #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #  source: https://pytorch.org/docs/stable/torchvision/models.html

    return tensor
    
    
def string_to_pilimage(pixelstring):
    imarray = pixelstring_to_numpy(pixelstring)
    out = pl.Image.fromarray(imarray).convert("L")
    return out


def tensor_to_pilimage(tensor, resolution = (256,256)):
    im = transforms.ToPILImage()(tensor.unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im


def create_datatensor_feedforward(dataset, sample = None):
    out = torch.tensor([], dtype = torch.uint8)
    for tensor in dataset["tensors"][:sample]:
        out = torch.cat((out, tensor.unsqueeze(0)))
    return out


def create_datatensor_vgg16(dataframe):
    """
    Stacks all the m tensors for each image contained in the dataframe into a single tensor.
    :return: tensor of shape (m, *shape). shape: shape of the tensors in dataframe["tensors"]
    For vgg16, shape is (3, 224, 224)
    """
    out = torch.stack(tuple(dataframe["tensors"].tolist()))
    return out


def label_to_vector(label):
    out = torch.zeros(emotions)
    out[label] = 1
    return out

