import numpy as np
import torch
import pywt
import pywt.data
import matplotlib.pyplot as plt
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.model_utils import write_dict, setup_model
from visualization.pltUtils import dict_from_file

def test_pywavelets():
    size_tensor = (64,64,64,32)

    feature_grid = torch.empty(size_tensor).uniform_(0, 1)

    # M: Testdata
    original = pywt.data.camera()  # M: ndarray

    # Method 1): Load image, Wavelet transform of image
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2

    # M: Method 2):
    coeffs = pywt.wavedec2(original, wavelet='db1', level=1)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # M: reconstruct
    #coeff_arr[0] = 0  #M: some filter
    coeffs_filt = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    reconstruction = pywt.waverec2(coeffs_filt, wavelet='db1')

    pass

def analyse_Wavelet():
    wavelet = pywt.Wavelet('db2')
    print(wavelet)

def test_TensorWavelets():
    size_tensor = (5, 16, 16, 16)
    feature_grid_orig = torch.empty(size_tensor).uniform_(0, 1)

    filter = WaveletFilter3d('db2')

    t = feature_grid_orig.shape[-3:]

    num_levels = None
    if num_levels is None:
        num_levels = min(pywt.dwt_max_level(s, filter.filter_length) for s in feature_grid_orig.shape[-3:])

    features = []  # M: wavelet coeffs representation of input feature grid
    shapes = []
    data = feature_grid_orig.detach().unsqueeze(0)
    for _ in range(num_levels):
        filtered, shape = filter.encode(data)
        features.append(filtered[0, :, 1:])
        shapes.append(shape)
        data = filtered[:, :, 0]  # M: only transform the lower part again

    features=[data[0]] + [*reversed(features)]
    shape_array = np.asarray(shapes[::-1], dtype=int)
    used_features = list([p for p in features])

    def decode_volume() -> torch.Tensor:
        restored = used_features[0].unsqueeze(0)
        for high_freq, shape in zip(used_features[1:], shape_array):
            data = torch.cat([restored.unsqueeze(2), high_freq.unsqueeze(0)], dim=2)
            restored = filter.decode(data, shape)
        return restored[0]

    decoded_feature_grid = decode_volume()
    pass


def analyse_coefficients():
    checkpoint_path = 'experiments/Tests/ImplTests/Finetuning/Test/model.pth'
    config_path = 'experiments/Tests/ImplTests/Finetuning/Test/config.txt'

    args = dict_from_file(config_path)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], '', args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])

    model.load_state_dict(torch.load(checkpoint_path))

    layers = []
    for name, layer in model.named_parameters():
        if 'feature_grid' in name:
            layers.append(layer.data.reshape(-1).cpu().numpy())

    fig, ax = plt.subplots(nrows=len(layers), ncols=1, figsize=(8, 8))
    fig.tight_layout()

    for i in range(len(layers)):
        ax[i].hist(layers[i], bins=160, label=str(i), range=(-0.5, 0.5))
        ax[i].title.set_text('Level ' + str(i))

    filepath = 'plots/Histogramms/' + 'test' + 'Feature_Historgramm_' + "Smallify" + '.png'
    plt.savefig(filepath)


if __name__ == '__main__':
    #test_pywavelets()
    #analyse_Wavelet()
    #test_TensorWavelets()
    analyse_coefficients()

