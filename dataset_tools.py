import PIL as pl
import numpy as np
import torch
import json
import torchvision.transforms as transforms

from sift_descriptors import SIFTDetector, DenseDetector

size = 48
emotions = 7

with open('config.json') as json_file:
    config = json.load(json_file)

# =============================================================================
# Conversion pixel tensor
# =============================================================================


def pixelstring_to_numpy(string, flatten=False, integer_pixels=False):
    pixels = string.split()
    if flatten:
        out = np.array([int(i) for i in pixels])
        return out
    out = np.zeros((size, size))
    for i in range(size):
        out[i] = np.array([int(k) for k in pixels[size * i:size * (i + 1)]])

    if integer_pixels:
        return out

    return out / 255.0


def descriptorstring_to_numpy(string):
    pixels = string.split()
    out = np.array([float(i) for i in pixels])
    return out


def pixelstring_to_torchtensor_feedforward(string, datatype=torch.float32, flatten=False, device=torch.device('cpu')):
    return torch.tensor(pixelstring_to_numpy(string, flatten=flatten), dtype=datatype).to(device)


def pixelstring_to_tensor_vgg16(pixels, device=torch.device('cpu')):
    # resize image to match input layer of vgg16
    img = string_to_pilimage(pixels).resize((224, 224))
    # input needs to have 3 channels
    tensor = transforms.ToTensor()(img).repeat((3, 1, 1))
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor).to(device)

    return tensor


def pixelstring_to_tensor_customvgg(pixels, device):
    return torch.tensor(pixelstring_to_numpy(pixels, flatten=False), dtype=torch.float32).unsqueeze_(0).to(device)


def pixelstring_batch_totensor(psb, pixelstring_to_tensor):
    out = torch.stack(tuple([pixelstring_to_tensor(string) for string in psb]))
    return out


def emotion_batch_totensor(emb, loss_mode="BCE"):
    if loss_mode == "BCE":
        return torch.stack(emb).T.float()
    else:
        return emb


# =============================================================================
# Concatenation des tenseurs
# =============================================================================

def create_datatensor_feedforward(dataset, sample=None, device=torch.device('cpu')):
    out = torch.tensor([], dtype=torch.float32).to(device)
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

def label_to_vector(label, device=torch.device('cpu')):
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


def tensor_to_pilimage(tensor, resolution=(256, 256)):
    im = transforms.ToPILImage()(tensor.unsqueeze_(0))
    im = transforms.Resize(resolution)(im)
    return im


# =============================================================================
# Pre-processing
# =============================================================================
def preprocess_batch_custom_vgg(pixelstring_batch, emotions_batch, DEVICE, with_data_aug=True, loss_mode="BCE"):
    transformations = [
        # pre-processing
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    if with_data_aug:
        transformations = [
                              # data augmentation
                              transforms.RandomHorizontalFlip(p=0.5)
                          ] + transformations

    pre_process = transforms.Compose(transformations)

    batch = torch.stack(
        tuple([
            pre_process(string_to_pilimage(string)) for string in pixelstring_batch
        ])
    )

    groundtruth = emotion_batch_totensor(emotions_batch, loss_mode)

    return batch, groundtruth


def preprocess_batch_hybrid(pixelstring_batch, emotions_batch, DEVICE, with_data_aug=True,
                            loss_mode="BCE"):

    transforms_flip = []
    if with_data_aug:
        transforms_flip = [transforms.RandomHorizontalFlip(p=0.5)]

    transforms_normalize = [
        # pre-processing
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[0.5])
    ]

    pre_process_flip = transforms.Compose(transforms_flip)
    pre_process_normalize = transforms.Compose(transforms_normalize)

    flipped_imgs = [pre_process_flip(string_to_pilimage(string)) for string in pixelstring_batch]

    detector = SIFTDetector()
    if config["sift_type"] == "dense":
        detector = DenseDetector()

    descriptors = [detector.compute_descriptors(np.array(im))[1] for im in flipped_imgs]
    descriptors = np.vstack([x.flatten() for x in descriptors if x is not None])

    descriptors_batch = torch.stack(tuple([torch.FloatTensor(d) for d in descriptors]))

    pixels_batch = torch.stack(tuple([
        pre_process_normalize(im) for im in flipped_imgs
    ]))

    groundtruth = emotion_batch_totensor(emotions_batch, loss_mode)

    return pixels_batch, descriptors_batch, groundtruth


def preprocess_batch_vgg16(pixelstring_batch, emotions_batch, DEVICE):
    groundtruth = emotion_batch_totensor(emotions_batch)
    batch = pixelstring_batch_totensor(pixelstring_batch, lambda x: pixelstring_to_tensor_vgg16(x, device=DEVICE))

    return batch, groundtruth


def preprocess_batch_feed_forward(pixelstring_batch, emotions_batch, DEVICE):
    groundtruth = emotion_batch_totensor(emotions_batch)
    batch = pixelstring_batch_totensor(pixelstring_batch,
                                       lambda x: pixelstring_to_torchtensor_feedforward(x, flatten=True, device=DEVICE))

    return batch, groundtruth
