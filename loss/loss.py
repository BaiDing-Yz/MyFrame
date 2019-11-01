import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

    
class My_MSE_loss(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale = scale
        
    def forward(self, x, y):
        delta = x - y
        scale = torch.clamp(2 *(torch.sign(delta) - 0.5),0,1) * self.scale
        return torch.sum(torch.pow(delta,2) * (scale + 1))
    
    

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7,scale = 5):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss()
        self.ce = My_MSE_loss(scale)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
    