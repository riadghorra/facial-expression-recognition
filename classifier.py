import torch
import torch.nn as nn
from torchvision import models


class FeedForwardNN(nn.Module):
    def __init__(self, n, hidden_sizes, device=torch.device('cpu')):
        super(FeedForwardNN, self).__init__()
        self.device = device
        self.first_couche = nn.Sequential(nn.Linear(n, hidden_sizes[0]), nn.ReLU()).to(self.device)
        self.hidden = [nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()).to(self.device) for i in range(len(hidden_sizes)-1)]
        self.last_couche = nn.Linear(hidden_sizes[-1],7).to(self.device)
        

    def forward(self, x):
        x = x.float()
        x = self.first_couche(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.last_couche(x)
        return x


def vgg16(device=torch.device('cpu')):
    """
    source: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    :return: vgg16 model with 7 output classes
    """

    vgg16 = models.vgg16(pretrained=True).to(device)

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False
        param = param.to(device)

    for param in vgg16.classifier.parameters():
        param.require_grad = True
        param = param.to(device)
    # replace last layer
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=7, bias=True).to(device)

    return vgg16
