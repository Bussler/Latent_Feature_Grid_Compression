import torch
import torch.nn as nn
import numpy as np


def SnakeAlt(x):
    return 0.5 * x + torch.sin(x) ** 2


class Feature_Grid_Model(nn.Module):
    def __init__(self, input_channel=31, hidden_channel=32, out_channel=1, num_layer=4):
        super(Feature_Grid_Model, self).__init__()

        self.input_channel = input_channel
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

        # M: interpolate feature entry

        # M: decode feature entry

        features_grid = torch.ones(16)  # M: TODO test data

        # M: enhance entry with fourier-features

        embedded_features = torch.empty(12).uniform_(0, 1)  # M: TODO test data

        x = torch.cat([input, embedded_features, features_grid], -1)

        # M: pass new x through NW
        for ndx, net_layer in enumerate(self.net_layers):
            # M: TODO: activation function!
            x = SnakeAlt(net_layer(x))

        x = self.final_layer(x)
        return x
