import os
import torch
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from model.Smallify_Dropout import SmallifyDropout
from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout, Straight_Through_Dropout
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d


def write_dict(dictionary, filename, experiment_path=''):
    with open(os.path.join(experiment_path, filename), 'w') as f:
        for key, value in dictionary.items():
            f.write('%s = %s\n' % (key, value))


def setup_model(input_channel, hidden_channel, out_channel, num_layer, embedding_type, n_embedding_freq, drop_type,
                drop_momentum, drop_threshold, wavelet_filter, grid_features, grid_size, checkpoint_path):

    # M: Setup Latent_Feature_Grid
    size_tensor = (grid_features, grid_size, grid_size, grid_size)
    feature_grid = torch.empty(size_tensor).uniform_(0, 1)

    # M: Setup wavelet_filter
    wavelet_filter = WaveletFilter3d(wavelet_filter)

    # M: Setup Drop-Layer for grid
    if drop_type:
        if drop_type == 'smallify':
            drop_layer = SmallifyDropout(feature_grid.shape[1:], drop_momentum, drop_threshold)
        if drop_type == 'straight_through':
            drop_layer = Straight_Through_Dropout(feature_grid.shape[1:], drop_momentum, drop_threshold)
        if drop_type == 'masked_straight_through':
            drop_layer = MaskedWavelet_Straight_Through_Dropout(feature_grid.shape[1:], drop_momentum, drop_threshold)
    else:
        drop_layer = None

    # M: Setup Embedder
    if embedding_type and embedding_type == 'fourier':
        embedder = FourierEmbedding(n_freqs=n_embedding_freq, input_dim=input_channel)
    else:
        embedder = FourierEmbedding(n_freqs=n_embedding_freq, input_dim=input_channel)

    # M: Setup model
    model = Feature_Grid_Model(embedder, feature_grid, drop_layer, wavelet_filter, input_channel_data=input_channel,
                               hidden_channel = hidden_channel, out_channel=out_channel, num_layer=num_layer)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model