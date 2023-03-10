import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import model.Feature_Grid_Model as feature_grid_model
from model.Dropout_Layer import DropoutLayer


# M: better to infer directly
def inference_variational_model(mu, sigma):
    return torch.normal(mu, sigma)


def calculate_Log_Likelihood(loss_criterion, predicted_volume, ground_truth_volume, log_sigma):
    x_mu_loss = loss_criterion(predicted_volume, ground_truth_volume)
    sigma = math.exp(log_sigma)
    a = 1 / (2 * (sigma ** 2))
    b = - (math.log(2 * math.pi) + (2 * log_sigma)) / 2

    return a * (-x_mu_loss) + b, x_mu_loss


def calculate_Log_Likelihood_variance(predicted_volume, ground_truth_volume, variance):
    x_mu_loss = (ground_truth_volume-predicted_volume) ** 2
    sigma = torch.exp(variance)
    a = 1 / (2 * (sigma ** 2))
    b = - (math.log(2 * np.pi) + (2 * variance)) / 2

    return a * (-x_mu_loss) + b, x_mu_loss


class VariationalDropoutLoss(nn.Module):

    def __init__(self, size_volume: float, batch_size: float, weight_dkl: float = 1., weight_weights: float = 1.,
                 weight_dkl_max=30.0):  # 30.0, 5
        super(VariationalDropoutLoss, self).__init__()
        self.batch_scale = (size_volume/batch_size)
        self.weight_dkl = float(weight_dkl)
        self.weight_dkl_max = weight_dkl_max
        self.weight_weights = float(weight_weights)
        self._reset_penalties()

    def _reset_penalties(self):
        self.DKL = []
        self.weight_loss = []

    def _collect_penalties(self, m: nn.Module):
        if isinstance(m, VariationalDropout):
            self.DKL.append(m.calculate_Dkl())
        if isinstance(m, feature_grid_model.Feature_Grid_Model):
            self.weight_loss.append(sum([torch.sum(torch.abs(f) ** 2) for f in m.feature_grid]))

    def forward(self, model: nn.Module, predicted_volume, ground_truth_volume, log_sigma, weight_dkl_multiplier):
        model.apply(self._collect_penalties)

        if self.weight_dkl < self.weight_dkl_max:
            self.weight_dkl = self.weight_dkl * (1.0 + weight_dkl_multiplier)

        Log_Likelyhood, mse = calculate_Log_Likelihood_variance(predicted_volume, ground_truth_volume, log_sigma)
        mse = mse.sum() * (1 / predicted_volume.shape[0])
        Log_Likelyhood = Log_Likelyhood.sum() * self.batch_scale
        Dkl_sum = self.weight_dkl * sum(self.DKL) * self.batch_scale
        weight_sum = self.weight_weights * sum(self.weight_loss) * self.batch_scale

        loss = -(Log_Likelyhood - Dkl_sum - weight_sum)

        self._reset_penalties()
        return loss, Log_Likelyhood, mse, Dkl_sum, weight_sum


def decode_variational_parameter(variational_layer, filter, shape_array):
    restored_thetas = variational_layer[0].log_thetas.unsqueeze(0).unsqueeze(0)
    for layer, shape in zip(variational_layer[1:], shape_array):
        high_freq = layer.log_thetas
        data = torch.cat([restored_thetas, high_freq.unsqueeze(0)], dim=1)
        restored_thetas = filter.decode(data.unsqueeze(0), shape)
    r_thetas = restored_thetas[0]

    restored_variances = variational_layer[0].log_var.unsqueeze(0).unsqueeze(0)
    for layer, shape in zip(variational_layer[1:], shape_array):
        high_freq = layer.log_var
        data = torch.cat([restored_variances, high_freq.unsqueeze(0)], dim=1)
        restored_variances = filter.decode(data.unsqueeze(0), shape)
    r_variances = restored_variances[0]

    thetas = torch.exp(r_thetas)  # M: revert the log with exp
    sigma = torch.exp(r_variances / 2.0)
    xi = torch.randn_like(thetas)  # M: draw xi from N(0,1)
    w = thetas + sigma * xi  # M: maybe have to unsqueeze(0)
    return w


class VariationalDropout(DropoutLayer):
    # M: constants from Molchanov variational dropout paper
    k1 = 0.63576
    k2 = 1.87320
    k3 = 1.48695
    C = -k1

    def __init__(self, size=(1,1,1), init_dropout=0.5, threshold=0.9):
        super(VariationalDropout, self).__init__(size, init_dropout, threshold)
        self.log_thetas = torch.nn.Parameter(torch.zeros(size), requires_grad=True)

        log_alphas = math.log(init_dropout / (1-init_dropout))
        # M: log_var = 2*log_sigma; sigma^2 = exp(2*log_sigma) = theta^2 alpha
        self.log_var = torch.nn.Parameter(torch.empty(size).fill_(log_alphas), requires_grad=True)

        self.d_mask = None

        #self.scaling = 1  # M: refactor!
        #if DropoutLayer.i == 2 or DropoutLayer.i == 3:
        #    self.scaling = 1/2
        #else: self.scaling = 1/3

    @property
    def alphas(self):
        return torch.exp(self.log_var - 2.0 * self.log_thetas)

    @property
    def dropout_rates(self):
        return self.alphas / (1.0 + self.alphas)  #M: maybe wrong? 1-alphas?

    @property
    def sigma(self):
        return torch.exp(self.log_var / 2.0)

    def forward(self, x):
        # M: w = theta * (1+sqrt(alpha)*xi)
        # M: w = theta + sigma * xi according to Molchanov additive noise reparamerization
        thetas = torch.exp(self.log_thetas)  # M: revert the log with exp
        #xi = torch.randn_like(x)  # M: draw xi from N(0,1)
        xi = torch.randn_like(thetas)
        w = thetas + self.sigma * xi  # M: maybe have to unsqueeze(0)

        if self.d_mask is None:
            return x * w
        else:
            return x * self.d_mask
        #return x

    def calculate_Dkl(self):
        log_alphas = self.log_var - 2.0 * self.log_thetas

        t1 = self.k1 * torch.sigmoid(self.k2 + self.k3 * log_alphas)
        t2 = 0.5 * F.softplus(-log_alphas, beta=1.)
        dkl = - t1 + t2 + self.k1

        return torch.sum(dkl)

    def calculate_Dropout_Entropy(self):
        drop_rate = self.dropout_rates
        h = drop_rate * torch.log(drop_rate) + (1.0 - drop_rate) * torch.log(1 - drop_rate)
        return torch.sum(h)

    def get_valid_fraction(self):
        not_dropped = torch.mean((self.dropout_rates < self.threshold).to(torch.float)).item()
        bin1 = torch.mean((self.dropout_rates < 0.1).to(torch.float)).item()
        bin3 = torch.mean((self.dropout_rates > 0.9).to(torch.float)).item()
        bin2 = 1.0-bin1-bin3
        return not_dropped, self.dropout_rates

    # M: If dropout_rates close to 1: alpha >> 1 and theta has no useful information
    def calculate_pruning_mask(self, device):
        with torch.no_grad():
            dropout_rates = self.dropout_rates
            prune_mask = torch.where(dropout_rates < self.threshold, 1.0, 0.0)

            if prune_mask.numel() - torch.count_nonzero(prune_mask) == 0:
                prune_mask.data[0] = 1.0

            self.d_mask = prune_mask.to(device) #M: store pruning mask, and after pruning only mult with this!

            return prune_mask.to(device)

    def multiply_values_with_dropout(self, input, device):
        with torch.no_grad():
            mask = self.calculate_pruning_mask(device) * torch.exp(self.log_thetas)
            f_grid = input * mask
            return f_grid

    def size_layer(self):
        return self.log_thetas.numel()


class Variance_Model(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, n_layers=4, size_layers=32):
        super(Variance_Model, self).__init__()

        self.net_layers = nn.ModuleList(
            [nn.Linear(input_ch, size_layers)] +
            [nn.Linear(size_layers, size_layers) for i in range(n_layers - 1)]
        )

        self.final_layer = nn.Linear(size_layers, output_ch)

    def forward(self, input):
        out = input
        for ndx, net_layer in enumerate(self.net_layers):
            out = F.relu(net_layer(out))
        out = self.final_layer(out)
        return out
