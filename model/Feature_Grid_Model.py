import torch
import torch.nn as nn
import numpy as np
import pywt
from model.Feature_Embedding import Embedder
from torch.nn import functional as F
from model.Dropout_Layer import DropoutLayer
from wavelet_transform.Torch_Wavelet_Transform import _WaveletFilterNd


def SnakeAlt(x):
    return 0.5 * x + torch.sin(x) ** 2


class Feature_Grid_Model(nn.Module):
    def __init__(self, embedder: Embedder, feature_grid, drop_layer: DropoutLayer, wavelet_filter: _WaveletFilterNd,
                 input_channel_data=3, hidden_channel=32, out_channel=1, num_layer=4):
        super(Feature_Grid_Model, self).__init__()

        self.embedder = embedder
        self.filter = wavelet_filter

        features, shapes = self.encode_volume(feature_grid)
        #self.feature_grid = torch.nn.Parameter(feature_grid, requires_grad=True)
        self.feature_grid = nn.ParameterList(values=[nn.Parameter(f, requires_grad=True) for f in features])
        self.shape_array = shapes

        self.drop = nn.ParameterList([drop_layer.create_instance(
            f.shape[1:], drop_layer.p, drop_layer.threshold) for f in features])

        self.input_channel = input_channel_data + embedder.out_dim + feature_grid.shape[0]
        self.hidden_width = hidden_channel
        self.output_channel = out_channel
        self.num_layer = num_layer

        self.net_layers = nn.ModuleList(
            [nn.Linear(self.input_channel, self.hidden_width)] +
            [nn.Linear(self.hidden_width, self.hidden_width) for i in range(self.num_layer-1)]
        )

        self.final_layer = nn.Linear(self.hidden_width, self.output_channel)

    def forward(self, input):

        # M: decode feature grid
        # feature_grid_samples = self.drop(self.feature_grid)
        feature_grid_samples = self.decode_volume()

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

    # M: creates wavelet representation of input feature grid
    def encode_volume(self, feature_volume, num_levels=None):
        if num_levels is None:
            num_levels = min(pywt.dwt_max_level(s, self.filter.filter_length) for s in feature_volume.shape[-3:])

        features = []  # M: wavelet coeffs representation of input feature grid
        shapes = []
        data = feature_volume.detach().unsqueeze(0)
        for _ in range(num_levels):
            filtered, shape = self.filter.encode(data)
            features.append(filtered[0, :, 1:])
            shapes.append(shape)
            data = filtered[:, :, 0]  # M: only transform the lower part (low freq, basic info) again

        features = [data[0]] + [*reversed(features)]
        shape_array = np.asarray(shapes[::-1], dtype=int)

        return features, shape_array

    # M: creates spatial representation of wavelet feature grid
    def decode_volume(self) -> torch.Tensor:
        restored = self.drop[0](self.feature_grid[0]).unsqueeze(0)
        for high_freq, drop_l, shape in zip(self.feature_grid[1:], self.drop[1:], self.shape_array):
            high_freq = drop_l(high_freq)
            data = torch.cat([restored.unsqueeze(2), high_freq.unsqueeze(0)], dim=2)
            restored = self.filter.decode(data, shape)
        return restored[0]

    def save_dropvalues_on_grid(self, device):
        masks = [d.calculate_pruning_mask(device) * d.betas for d in self.drop]
        f_grid = [grid * mask for grid, mask in zip(self.feature_grid, masks)]
        self.feature_grid = nn.ParameterList(values=[nn.Parameter(f, requires_grad=True) for f in f_grid])
