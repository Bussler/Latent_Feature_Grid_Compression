import torch
import torch.nn as nn
from torch.nn.functional import linear
import numpy as np
from model.Dropout_Layer import DropoutLayer
from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout, Straight_Through_Dropout
from model.Feature_Grid_Model import Feature_Grid_Model


class SmallifyLoss(nn.Module):

    def __init__(self, weight_l1: float = 1., weight_l2: float = 1.):
        super(SmallifyLoss, self).__init__()
        self.weight_l1 = float(weight_l1)
        self.weight_l2 = float(weight_l2)
        self._reset_penalties()

    def _reset_penalties(self):
        self._penalties_l1 = []
        self._penalties_l2 = []

    def _collect_penalties(self, m: nn.Module):
        if isinstance(m, SmallifyDropout):
            self._penalties_l1.append(m.l1_loss())
        if isinstance(m, MaskedWavelet_Straight_Through_Dropout):
            self._penalties_l1.append(m.l1_loss())
        if isinstance(m, Straight_Through_Dropout):
            self._penalties_l1.append(m.l1_loss())
        if isinstance(m, Feature_Grid_Model):
            self._penalties_l2.append(sum([torch.sum(torch.abs(f) ** 2) for f in m.feature_grid]))

    def forward(self, model: nn.Module) -> torch.Tensor:
        model.apply(self._collect_penalties)
        loss = 0.
        if self.weight_l1 > 0.:
            loss = loss + self.weight_l1 * sum(self._penalties_l1)
        if self.weight_l2 > 0.:
            loss = loss + self.weight_l2 * sum(self._penalties_l2)
        self._reset_penalties()
        return loss


class SmallifyDropout(DropoutLayer):

    def __init__(self, size=(1,1,1), sign_variance_momentum=0.025, threshold=0.75):
        super(SmallifyDropout, self).__init__(size, sign_variance_momentum, threshold)
        self.betas = torch.nn.Parameter(torch.empty(size).normal_(0, 1),
                                        requires_grad=True)  # M: uniform_ or normal_
        #self.betas = torch.nn.Parameter(torch.ones(size),
        #                                requires_grad=True)
        self.tracker = SmallifySignVarianceTracker(self.c, sign_variance_momentum, self.threshold, self.betas)
        self.d_mask = None

    def forward(self, x):
        if self.training:
            if self.d_mask is None:
                x = x.mul(self.betas.unsqueeze(0))  # M: No inverse scaling needed here, since we mult betas with nw after training
                self.tracker.sign_variance_pruning_onlyVar(self.betas)
            else:
                x = x * self.d_mask.unsqueeze(0)
        return x

    def l1_loss(self):
        return torch.abs(self.betas).sum()

    def calculate_pruning_mask(self, device):
        mask = self.tracker.calculate_pruning_mask(device)
        self.d_mask = mask  # M: store pruning mask, and after pruning only mult with this!
        return mask

    def multiply_values_with_dropout(self, input, device):
        with torch.no_grad():
            mask = self.calculate_pruning_mask(device) * self.betas.unsqueeze(0)
            f_grid = input * mask
            return f_grid

    def size_layer(self):
        return self.betas.numel()


class SmallifySignVarianceTracker():
    def __init__(self, c, sign_variance_momentum, threshold, betas):
        self.c = c
        self.sign_variance_momentum = sign_variance_momentum
        self.EMA, self.EMAVar = self.init_variance_data(betas)
        self.threshold = threshold

    def init_variance_data(self, betas):
        EMA = torch.sign(betas)
        EMAVar = torch.zeros(self.c)

        return EMA, EMAVar

    def sign_variance_pruning(self, device, betas):
        with torch.no_grad():
            newVal = torch.sign(betas).cpu()
            phi_i = newVal - self.EMA
            self.EMA = self.EMA + (self.sign_variance_momentum * phi_i)
            self.EMAVar = (torch.ones(self.c) - self.sign_variance_momentum) * \
                             (self.EMAVar + (self.sign_variance_momentum * (phi_i ** 2)))

            prune_mask = torch.where(self.EMAVar < self.threshold, 1.0, 0.0)  #M: extend to set betas directly to 0

        return prune_mask.to(device)

    def sign_variance_pruning_onlyVar(self, betas):
        with torch.no_grad():
            newVal = torch.sign(betas).cpu()
            phi_i = newVal - self.EMA
            self.EMA = self.EMA + (self.sign_variance_momentum * phi_i)
            self.EMAVar = (torch.ones(self.c) - self.sign_variance_momentum) * \
                             (self.EMAVar + (self.sign_variance_momentum * (phi_i ** 2)))

    def calculate_pruning_mask(self, device):
        with torch.no_grad():
            prune_mask = torch.where(self.EMAVar < self.threshold, 1.0, 0.0)

        return prune_mask.to(device)
