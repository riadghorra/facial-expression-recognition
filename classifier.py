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


class Custom_vgg(nn.Module):
    def __init__(self,in_channel, out_dim, device=torch.device('cpu')):
        super(Custom_vgg, self).__init__()
        self.device = device
        self.convs1 = nn.Sequential(nn.Conv2d(in_channel,32, 3,padding=1),nn.ReLU(True),
                                    nn.Conv2d(32,32, 3,padding=1), nn.ReLU(True)).to(self.device)
        
        self.pool1 = nn.MaxPool2d(2, stride=2).to(self.device)
        self.convs2 = nn.Sequential(nn.Conv2d(32,64, 3,padding=1), nn.ReLU(True),
                                    nn.Conv2d(64,64, 3,padding=1), nn.ReLU(True)).to(self.device)
        
        self.pool2 = nn.MaxPool2d(2, stride=2).to(self.device)
        self.convs3 = nn.Sequential(nn.Conv2d(64,128, 3,padding=1), nn.ReLU(True), 
                                    nn.Conv2d(128,128, 3,padding=1), nn.ReLU(True), 
                                    nn.Conv2d(128,128, 3,padding=1), nn.ReLU(True)).to(self.device)
        
        self.pool3 =  nn.MaxPool2d(2, stride=2).to(self.device)
        self.flat = torch.nn.Flatten().to(self.device)
        self.FC512 = nn.Sequential(nn.Linear(4608, 512), nn.ReLU(True)).to(self.device)
        self.FC256 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True)).to(self.device)
        self.FCOUT = nn.Sequential(nn.Linear(256, out_dim)).to(self.device)
    
    def forward(self,x):
        x = self.convs1(x)
        x = self.pool1(x)
        x = self.convs2(x)
        x = self.pool2(x)
        x = self.convs3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.FC512(x)
        x = self.FC256(x)
        x = self.FCOUT(x)
        return x
    
    def readable_output(self, x, cats):
        softmax = nn.Softmax(dim=1).to(self.device)
        y = softmax(self.forward(x))[0]
        for i, cat in enumerate(cats):
            print("Le visage appartient à la categorie {} à {}%".format(cat, round(float(100 * y[i]), 2)))

    def predict_single(self, x):
        softmax = nn.Softmax(dim=1).to(self.device)
        y = softmax(self.forward(x))[0]
        return [round(float(100*proba), 2) for proba in y]
