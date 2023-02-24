import torch
import torch.nn as nn
from torch.nn.functional import linear
import numpy as np
from model.Dropout_Layer import DropoutLayer
from model.Feature_Grid_Model import Feature_Grid_Model
from torch.nn import functional as F

# M: Own Implementation
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        return input < thresh

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class Straight_Through_Dropout(DropoutLayer):

    def __init__(self, size=(1,1,1), probability=0.5, threshold=0.5):
        super(Straight_Through_Dropout, self).__init__(size, probability, threshold)
        self.mask_values = torch.nn.Parameter(torch.ones(size), requires_grad=True)  # M: uniform_ or normal_

    def forward(self, x):
        if self.training:
            binary_mask = STEFunction.apply(torch.rand(self.c, device=x.device), self.mask_values)
            x = x.mul(binary_mask.unsqueeze(0))  # M: Inverse scaling needed?
        return x

    def l1_loss(self):
        return torch.abs(self.mask_values).sum()

    def calculate_pruning_mask(self, device):
        return self.mask_values > self.threshold

    def multiply_values_with_dropout(self, input, device):
        with torch.no_grad():
            mask = self.calculate_pruning_mask(device)
            f_grid = input * mask
            return f_grid


# M: Implementation as descirbed by Masked Wavelet Representation paper
class MaskedWavelet_Straight_Through_Dropout(DropoutLayer):

    def __init__(self, size=(1, 1, 1), probability=0.5, threshold=0.5):
        super(MaskedWavelet_Straight_Through_Dropout, self).__init__(size, probability, threshold)
        self.mask_values = torch.nn.Parameter(torch.ones(size), requires_grad=True)  # M: uniform_ or normal_
        self.d_mask = None

    def forward(self, x):
        if self.training:
            mask = torch.sigmoid(self.mask_values)

            if self.d_mask is None:
                x = (x * (mask >= self.threshold) - x * mask).detach() + (x * mask)  # M: Need inverse scaling?
            else:
                x = x * self.d_mask
        return x

    def l1_loss(self):
        return torch.abs(self.mask_values).sum()

    def calculate_pruning_mask(self, device):
        mask = torch.sigmoid(self.mask_values)
        self.d_mask = (mask >= self.threshold).to(device)  # M: store pruning mask, and after pruning only mult with this!
        return mask

    def multiply_values_with_dropout(self, input, device):
        with torch.no_grad():
            mask = self.calculate_pruning_mask(device)
            f_grid = (input * (mask >= self.threshold) - input * mask).detach() + (input * mask)
            return f_grid

    def size_layer(self):
        return self.mask_values.numel()
