import torch 
import torch
from torch import nn

class FullConnect(nn.Module):
    def __init__(self, n_input, limit_f, n_output):
        super(FullConnect, self).__init__()
        self.relu_act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.model = []

        while n_input > limit_f:
            o_ft = int(n_input / 2)
            self.model.append(nn.Linear(in_features=n_input, out_features=o_ft))
            self.model.append(self.relu_act)
            n_input = o_ft
        self.model.append(nn.Linear(in_features=n_input, out_features=n_output))
        self.model = nn.Sequential(*self.model)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
        
