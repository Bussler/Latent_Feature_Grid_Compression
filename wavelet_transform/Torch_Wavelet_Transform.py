from typing import List, Union, Optional

import numpy as np
import pywt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam


def _as_wavelet(wavelet: Union[str, pywt.Wavelet]):
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    return pywt.Wavelet(wavelet)

# M: Taken from Kevin Hoehlein https://github.com/khoehlein
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


class WaveletFilter3d(_WaveletFilterNd):

    def __init__(self, wavelet: Union[str, pywt.Wavelet], padding='constant'):
        super(WaveletFilter3d, self).__init__(wavelet, 3, F.conv3d, F.conv_transpose3d, padding=padding)