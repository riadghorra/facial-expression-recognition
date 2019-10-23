import torch
import torch.nn as nn
import torch.utils
import torch.utils.data


class classifier(nn.Module):
    def __init__(self, n, hidden_sizes):
        super(classifier,self).__init__()
        self.first_couche = nn.Sequential(nn.Linear(n, hidden_sizes[0]), nn.ReLU())
        self.hidden = [nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()) for i in range(len(hidden_sizes)-1)]
        self.last_couche = nn.Linear(hidden_sizes[-1],7)
        
    def forward(self, x):
        x = x.float()
        x = self.first_couche(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.last_couche(x)
        return x
