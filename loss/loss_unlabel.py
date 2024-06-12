from __future__ import print_function, division
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import numpy as np

class MSELoss_unlabel(_Loss):
    def __init__(self):
        super(MSELoss_unlabel, self).__init__()

    def forward(self, input_y, target, weight):
        # assert target.requires_grad is False
        weight = weight.float()
        target = target.float()
        loss = weight * ((input_y - target) ** 2)
        loss = torch.sum(loss) / torch.sum(weight)
        return loss


class BCELoss_unlabel(_Loss):
    def __init__(self):
        super(BCELoss_unlabel, self).__init__()

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        input_y = torch.clamp(input_y, min=0.000001, max=0.999999)
        weight = weight.float()
        target = target.float()
        loss = -weight* (target * torch.log(input_y) + (1 - target) * torch.log(1 - input_y))
        loss = torch.sum(loss) / torch.sum(weight)
        return loss
