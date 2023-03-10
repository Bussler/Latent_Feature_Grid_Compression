from collections import namedtuple
from itertools import chain
from typing import List, Union, Optional

import numpy as np
import pywt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


def _as_wavelet(wavelet: Union[str, pywt.Wavelet]):
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    return pywt.Wavelet(wavelet)


class _WaveletFilterNd(nn.Module):
    """
    Base class for n-dimensional wavelet filters
    """

    def __init__(
            self, wavelet: pywt.Wavelet,
            dim: int, conv_forward, conv_reverse,
            padding='constant'
    ):
        super(_WaveletFilterNd, self).__init__()
        self.dim = dim
        self._conv_forward = conv_forward # n-dim convolution function
        self._conv_reverse = conv_reverse # n-dim transpose convolution function
        self.padding = padding # padding mode tobe used in convolution
        self._register_filters(_as_wavelet(wavelet), dim)  # M: build filters for wavelet transform and inverse
        assert self.filter_length % 2 == 0, '[ERROR] Implementation does not support uneven filter length'

    @property
    def filter_length(self):
        return self.filter_fwd.shape[-1]

    def _register_filters(self, wavelet: pywt.Wavelet, dim: int):
        # get 1-dim wavelet kernels from pywavelet object
        fwd_low, fwd_high, rev_low, rev_high = (torch.tensor(x) for x in wavelet.filter_bank)

        # generate n-dim wavelet kernels via iterative outer products
        def build_next_filter_bank(filters_1d: List[Tensor], filters_nd: List[Tensor]):
            filters_1d = [f.unsqueeze(-1) for f in filters_1d]
            out = [f1 * fn for f1 in filters_1d for fn in filters_nd]
            return filters_1d, out

        def build_ndim_filter(filters_1d: List[Tensor]):
            out = [f.unsqueeze(0) for f in filters_1d]
            for _ in range(1, dim):
                filters_1d, out = build_next_filter_bank(filters_1d, out)
            return torch.stack(out, dim=0).unsqueeze(1)

        # register n-dim kernels as buffers of the module
        self.register_buffer('filter_fwd', build_ndim_filter([fwd_low.flip(-1), fwd_high.flip(-1)]))
        self.register_buffer('filter_rev', build_ndim_filter([rev_low, rev_high]))

    def _get_padding_size(self, shape: np.ndarray):
        is_odd = shape % 2 == 1
        out = np.full(6, (2 * self.filter_length - 3) // 2, dtype=int)
        out[1::2] += is_odd.astype(int)
        return tuple(out)

    def _pad_for_forward(self, data: Tensor):
        shape = np.asarray(data.shape[-self.dim:])
        return F.pad(data, self._get_padding_size(shape), mode=self.padding),shape

    def _unpad_for_reverse(self, data: Tensor, shape: np.ndarray):
        difference = np.asarray(data.shape[-self.dim:]) - shape
        slices = [slice(int(np.floor(d / 2)), -int(np.ceil(d / 2)) or None, None) for d in difference]
        slices = [slice(None, None, None), slice(None, None, None)] + slices
        return data[slices]

    def encode(self, data: Tensor):
        """
        Forward wavelet transform
        :param data: data tensor of shape (batch, channel, *spatial size)
        :return coeffs: Tensor of wavelet coefficients, shape (batch, channels, coeffs, *spatial size // 2)
        :return shape: Tuple of size of input tensor (needed for padding during reconstruction)
        """
        assert len(data.shape) == 2 + self.dim, \
            f'[ERROR] WaveletFilter{self.dim}d.forward(...) expects input of dimension {self.dim + 2}. Got {len(data.shape)} instead.'
        data, shape = self._pad_for_forward(data)
        batch_size, num_channels, *_ = data.size()
        weight = self.filter_fwd.repeat(num_channels, *[1 for _ in range(self.dim + 1)])
        result = self._conv_forward(data, weight, stride=2, groups=num_channels)
        coeffs = result.reshape(batch_size, num_channels, 2 ** self.dim, *result.shape[2:])
        return coeffs, shape

    def decode(self, data: Tensor, shape: np.ndarray):
        """
        Reverse wavelet transform
        :param data: Tensor of wavelet coefficients of shape (batch, channels, coeffs, *spatial size)
        :param shape: Desired output shape as obtained from encode function
        :return result: restored data of shape (batch, channels, *(2x spatial size))
        """
        assert len(data.shape) == 3 + self.dim, \
            f'[ERROR] WaveletFilter{self.dim}d.reverse(...) expects input of dimension {self.dim + 3}. Got {len(data.shape)} instead.'
        batch_size, num_channels, *_ = data.shape
        weight = self.filter_rev.repeat(num_channels, *[1 for _ in range(self.dim + 1)])
        result = self._conv_reverse(torch.flatten(data, start_dim=1, end_dim=2), weight, groups=num_channels, stride=2)
        result = self._unpad_for_reverse(result, shape)
        return result

    def forward(self, data: Tensor):
        return self.encode(data)[0]


class WaveletFilter1d(_WaveletFilterNd):

    def __init__(self, wavelet: Union[str, pywt.Wavelet], padding='constant'):
        super(WaveletFilter1d, self).__init__(wavelet, 1, F.conv1d, F.conv_transpose1d, padding=padding)


class WaveletFilter2d(_WaveletFilterNd):

    def __init__(self, wavelet: Union[str, pywt.Wavelet], padding='constant'):
        super(WaveletFilter2d, self).__init__(wavelet, 2, F.conv2d, F.conv_transpose2d, padding=padding)


class WaveletFilter3d(_WaveletFilterNd):

    def __init__(self, wavelet: Union[str, pywt.Wavelet], padding='constant'):
        super(WaveletFilter3d, self).__init__(wavelet, 3, F.conv3d, F.conv_transpose3d, padding=padding)


class MultilevelVolumeFeatures(nn.Module):

    @classmethod
    def from_tensor(cls, data: Tensor, filter: WaveletFilter3d, num_levels: Optional[int] = None):
        assert len(data.shape) == 4 # expected shape (num_channels, H, W, D)
        if num_levels is None:
            num_levels = min(pywt.dwt_max_level(s, filter.filter_length) for s in data.shape[-3:])  # M wie hier die wavelet level anzahl berechnet wird, verstehe ich noch nicht
        features = []  # M: wavelet coeffs representation of input feature grid
        shapes = []
        data = data.detach().unsqueeze(0)
        for _ in range(num_levels):
            filtered, shape = filter.encode(data)
            features.append(filtered[0, :, 1:])
            shapes.append(shape)
            data = filtered[:, :, 0] # M: only transform the lower part again
        return cls(filter, np.asarray(shapes[::-1], dtype=int), data[0], *reversed(features))

    def forward(self, x: Tensor) -> Tensor:
        grid = x.view(1, 1, 1, *x.shape)
        samples = F.grid_sample(self.decode_volume().unsqueeze(0), grid, mode='bilinear', align_corners=False)
        samples = samples.view(-1, x.shape[0]).transpose(0, 1)
        return samples

    def __init__(self, filter: WaveletFilter3d, shapes: np.ndarray, *features: Tensor):
        # expects features[0] to be the lowest-order low-pass features
        super(MultilevelVolumeFeatures, self).__init__()
        self.filter = filter
        self._verify_features(features)
        #self.features = nn.ParameterList(parameters=[nn.Parameter(p, requires_grad=True) for p in features])  # M: this is the latent feature grid
        self.features = nn.ParameterList(values=[nn.Parameter(p, requires_grad=True) for p in features])
        self._verify_shapes(shapes)
        self.shapes = shapes
        self.to(features[0].device)

    @staticmethod
    def _verify_features(features):
        assert len(features) > 0
        # check lowest-level average features
        assert len(features[0].shape) == 4  # expected shape: (out_channels, H, W, D)
        out_channels = features[0].shape[0]
        # check detail coefficients
        for p in features[1:]:
            assert len(p.shape) == 5  # expected shape: (out_channels, 7, H', W', D')
            assert p.shape[0] == out_channels
            assert p.shape[1] == 7 # number of detail coefficients

    def _verify_shapes(self, shapes):
        assert len(shapes) == self.num_levels
        assert shapes.shape[-1] == 3

    @property
    def num_levels(self) -> int:
        return len(self.features) - 1

    @property
    def out_channels(self) -> int:
        return self.features[0].shape[0]

    def decode_volume(self) -> Tensor:
        restored = self.features[0].unsqueeze(0)
        for high_freq, shape in zip(self.features[1:], self.shapes):
            data = torch.cat([restored.unsqueeze(2), high_freq.unsqueeze(0)], dim=2)
            restored = self.filter.decode(data, shape)
        return restored[0]


# M: NW only learning on Feature Grid and mask!
# M: TODO in my NW: Decode Features from MW before or after passing to MLP? -> Denke mal vorher!
class MultilevelSmallifyFeatures(MultilevelVolumeFeatures):

    def __init__(self, filter: WaveletFilter3d, shapes: np.ndarray, *features: Tensor):
        super(MultilevelSmallifyFeatures, self).__init__(filter, shapes, *features)
        self.betas = nn.ParameterList([nn.Parameter(torch.ones_like(p[0])) for p in self.features])
        pass

    def decode_volume(self) -> Tensor:
        restored = (self.features[0] * self.betas[0].unsqueeze(0)).unsqueeze(0)
        for high_freq, betas, shape in zip(self.features[1:], self.betas[1:], self.shapes):
            rescaled = high_freq * betas.unsqueeze(0)
            data = torch.cat([restored.unsqueeze(2), rescaled.unsqueeze(0)], dim=2)
            restored = self.filter.decode(data, shape)
        return restored[0]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._update_trackers()
        return super(MultilevelSmallifyFeatures, self).forward(x)

    def _update_trackers(self):
        pass


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
        if isinstance(m, MultilevelSmallifyFeatures):
            self._penalties_l1.append(sum([torch.sum(torch.abs(p)) for p in m.betas]))
            self._penalties_l2.append(sum([torch.sum(torch.abs(p) ** 2) for p in m.features]))

    def forward(self, model: nn.Module) -> Tensor:
        model.apply(self._collect_penalties)
        loss = 0.
        if self.weight_l1 > 0.:
            loss = loss + self.weight_l1 * sum(self._penalties_l1)
        if self.weight_l2 > 0.:
            loss = loss + self.weight_l2 * sum(self._penalties_l2)
        self._reset_penalties()
        return loss


def _test():

    from torch.utils.data import TensorDataset, DataLoader

    def build_test_volume(num_centers=3, resolution=16):
        centers = torch.randn(num_centers, 3)
        nodes = torch.linspace(-1, 1, resolution)
        grid = torch.stack(torch.meshgrid(nodes, nodes, nodes, indexing='ij'), dim=-1).flatten(end_dim=-2)
        values = torch.exp(- (grid.unsqueeze(0) - centers.unsqueeze(1)).sum(dim=-1, keepdim=True) ** 2 / 2.).sum(dim=0)
        return grid, values

    def build_model(resolution=16,features=None):  #latent_features hier als 16^3, 1 feld -> bei mir dann
        filter = WaveletFilter3d('db2')
        if features is None:
            features = (2. * torch.rand(1, resolution, resolution, resolution) - 1.) * 0.05
        model = MultilevelSmallifyFeatures.from_tensor(features, filter)
        return model

    def disable_feature_update(m: nn.Module):
        if isinstance(m, MultilevelSmallifyFeatures):
            for p in m.features:
                p.requires_grad = False

    # build target scalar field
    grid, values = build_test_volume()  # normalized pos grid: (4096, 3); scalar values: (4096, 1)
    volume = torch.reshape(values, (1, 1, 16, 16, 16))

    def _sample_volume(positions):
        """
        function for sampling the reference field
        :param positions: 3D position tensor of shape (batch, 3) with values in [-1, 1]^3, as used in the grid convention of F.grid_sample(...)
        :return samples: tensor of volume samples of shape (batch, 1)
        """
        samples = F.grid_sample(volume, positions.view(1, 1, 1, *positions.shape))
        samples = samples.view(-1, 1)
        return samples

    loader = DataLoader(TensorDataset(grid, values), batch_size=1024)
    # model is initialized to equal the target field for faster fitting
    model = build_model(features=torch.reshape(values[:, 0], (16, 16, 16, 1)).T)
    train_features = True # allows update of feature volume for better fitting
    # For mask-only training:
    # train_features = False

    # For random initialization:
    # model = build_model(resolution=16)
    # train_features = True

    mse_loss = nn.MSELoss()
    smallify_loss = SmallifyLoss(weight_l1=1.e-6, weight_l2=0.)  # M: sparsity in der mask!
    optimizer = Adam(model.parameters(), lr=1.e-2)
    # scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    num_epochs = 1000


    def train():
        model.train()
        if not train_features:
            model.apply(disable_feature_update)
        for i, batch in enumerate(loader):
            model.zero_grad()
            positions, _ = batch
            targets = _sample_volume(positions)
            samples = model(positions)
            loss = mse_loss(samples, targets) + smallify_loss(model)
            loss.backward()
            optimizer.step()

    def validate(message):
        model.eval()
        with torch.no_grad():
            samples = model(grid)
            targets = _sample_volume(grid)
            rmse = torch.sqrt(mse_loss(samples, targets))
            penalty = smallify_loss(model)
            print(f'[INFO] {message}: RMSE = {rmse.item()}, Smallify = {penalty}')
        return rmse.item() ** 2 + penalty

    validate('Before training')
    for epoch in range(num_epochs):
        train()
        loss = validate(f'After epoch {epoch + 1}')
        # scheduler.step(loss)


    print('[INFO] Finished')
    print([torch.mean(torch.abs(b)) for b in model.betas])
    print([torch.min(torch.abs(b)) for b in model.betas])
    print([torch.max(torch.abs(b)) for b in model.betas])



if __name__ == '__main__':
    _test()