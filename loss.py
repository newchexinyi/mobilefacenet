import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, target):
        loss = self.ce(x, target)
        return loss