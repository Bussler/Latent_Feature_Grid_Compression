import torch
import torch.nn as nn
import numpy as np
from model.Feature_Embedding import Embedder
from torch.nn import functional as F
from model.Dropout_Layer import DropoutLayer


def SnakeAlt(x):
    return 0.5 * x + torch.sin(x) ** 2


class Feature_Grid_Model(nn.Module):
    def __init__(self, embedder: Embedder, feature_grid, drop_layer: DropoutLayer,
                 input_channel_data=3, hidden_channel=32, out_channel=1, num_layer=4):
        super(Feature_Grid_Model, self).__init__()

        self.embedder = embedder
        self.feature_grid = torch.nn.Parameter(feature_grid, requires_grad=True)
        self.drop = drop_layer

        self.input_channel = input_channel_data + embedder.out_dim + self.feature_grid.shape[0]
        self.hidden_width = hidden_channel
        self.output_channel = out_channel
        self.num_layer = num_layer

        self.net_layers = nn.ModuleList(
            [nn.Linear(self.input_channel, self.hidden_width)] +
            [nn.Linear(self.hidden_width, self.hidden_width) for i in range(self.num_layer-1)]
        )

        self.final_layer = nn.Linear(self.hidden_width, self.output_channel)

    def forward(self, input):

        # M: multiply feature grid with mask
        feature_grid_samples = self.drop(self.feature_grid)  # M: TODO: look that values are mult correctly!

        # M: TODO decode feature grid

        # M: interpolate feature entry
        if not self.training:  # M: TODO refactor this
            orig_shape = input.shape
            input = input.squeeze()
            input = input.view(input.shape[0]*input.shape[1]*input.shape[2], input.shape[3])

        grid = input.view(1, 1, 1, *input.shape)
        feature_grid_samples = F.grid_sample(feature_grid_samples.unsqueeze(0), grid,
                                             mode='bilinear', align_corners=False).squeeze().transpose_(0, 1)

        # M: enhance entry with fourier-features
        embedded_features = self.embedder.embed(input)

        x = torch.cat([input, embedded_features, feature_grid_samples], -1) #feature_grid_samples

        # M: pass new x through NW
        for ndx, net_layer in enumerate(self.net_layers):
            x = SnakeAlt(net_layer(x))

        x = self.final_layer(x)

        if not self.training:
            x = x.view(orig_shape[0:-1],1)

        return x

    def save_dropvalues_on_grid(self, device):
        mask = self.drop.calculate_pruning_mask(device) * self.drop.betas
        f_grid = self.feature_grid * mask
        self.feature_grid = torch.nn.Parameter(f_grid, requires_grad=True)
