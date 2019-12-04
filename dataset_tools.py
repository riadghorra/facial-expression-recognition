import PIL as pl
import numpy as np
import torch
import torchvision.transforms as transforms

size = 48
emotions = 7

# =============================================================================
# Conversion pixel tensor
# =============================================================================


def pixelstring_to_numpy(string, flatten = False, integer_pixels=False):
    pixels = string.split()
    if flatten :
        out = np.array([int(i) for i in pixels])
        return out
    out = np.zeros((size,size))
    for i in range(size):
        out[i] = np.array([int(k) for k in pixels[size*i:size*(i+1)]])

    if integer_pixels:
        return out

    return out/255.0


def pixelstring_to_torchtensor_feedforward(string, datatype = torch.float32, flatten = False, device = torch.device('cpu')):
    return torch.tensor(pixelstring_to_numpy(string, flatten = flatten), dtype = datatype).to(device)


def pixelstring_to_tensor_vgg16(pixels, device = torch.device('cpu')):
    # resize image to match input layer of vgg16
    img = string_to_pilimage(pixels).resize((224, 224))
    # input needs to have 3 channels
    tensor = transforms.ToTensor()(img).repeat((3, 1, 1))
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor).to(device)

    return tensor

def pixelstring_to_tensor_customvgg(pixels, device):
    return torch.tensor(pixelstring_to_numpy(pixels, flatten = False), dtype = torch.float32).unsqueeze_(0).to(device)

def pixelstring_batch_totensor(psb, pixelstring_to_tensor):
    out = torch.stack(tuple([pixelstring_to_tensor(string) for string in psb]))
    return out

def emotion_batch_totensor(emb):
    out = torch.stack(tuple([label_to_vector(em) for em in emb]))
    return out

    

# =============================================================================
# Concatenation des tenseurs
# =============================================================================

def create_datatensor_feedforward(dataset, sample = None, device = torch.device('cpu')):
    out = torch.tensor([], dtype = torch.float32).to(device)
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

# =============================================================================
# One-hot encoding
# =============================================================================

def label_to_vector(label, device = torch.device('cpu')):
    out = torch.zeros(emotions).to(device)
    out[label] = 1
    return out


# =============================================================================
# Visualisation
# =============================================================================

def string_to_pilimage(pixelstring):
    imarray = pixelstring_to_numpy(pixelstring, integer_pixels=True)
    out = pl.Image.fromarray(imarray).convert("L")
    return out


def tensor_to_pilimage(tensor, resolution = (256,256)):
    im = transforms.ToPILImage()(tensor.unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im
