import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.l1 = nn.Linear(784, 300)
        self.l2 = nn.Linear(300,100)
        self.l3 = nn.Linear(100,10)
        self.relu = nn.ReLU(inplace = True)


    def forward(self, x):

        x = torch.flatten(x,start_dim = 1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)

        return x

