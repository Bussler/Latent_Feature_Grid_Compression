import os
import torch
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from model.Smallify_Dropout import SmallifyDropout
from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout, Straight_Through_Dropout
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.Variational_Dropout_Layer import VariationalDropout
import struct
from sklearn.cluster import KMeans


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
        if 'variational' in drop_type:
            drop_layer = VariationalDropout(feature_grid.shape[1:], drop_momentum, drop_threshold)
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


def get_net_weights_biases(net):
    weights = []
    biases = []
    for name, param in net.named_parameters():  # M: weight matrices end with .weight; biases with .bias
        if name.endswith('.weight'):
            weights.append(param.data)
        if name.endswith('.bias'):
            biases.append(param.data)
    return weights, biases


def store_model_parameters(model, filename):
    # M: basic information
    file = open(filename, 'wb')

    n_layers = model.num_layer
    layer_width = model.hidden_width
    input_dim = model.input_channel
    output_dim = model.output_channel

    n_grids = len(model.feature_grid)
    feature_size = model.feature_grid[0].shape[0]
    grid_dims = []
    for grid in model.feature_grid:
        grid_dims.append(len(grid.shape)-1)
    grid_sizes = []
    for grid in model.feature_grid:
        grid_sizes.append(grid.shape[1:])

    # M: header
    file.write(struct.pack('B', n_layers))
    file.write(struct.pack('B', layer_width))
    file.write(struct.pack('B', input_dim))
    file.write(struct.pack('B', output_dim))

    file.write(struct.pack('B', n_grids))
    file.write(struct.pack('B', feature_size))
    file.write(struct.pack(''.join(['I' for _ in range(len(grid_dims))]), *grid_dims))
    for grid in grid_sizes:
        file.write(struct.pack(''.join(['I' for _ in range(len(grid))]), *grid))

    # M: model params
    net_weights, net_biases = get_net_weights_biases(model)
    for cur_weigth, cur_bias in zip(net_weights, net_biases):
        cur_weigth = cur_weigth.view(-1).tolist()
        weight_format = ''.join(['f' for _ in range(len(cur_weigth))])
        file.write(struct.pack(weight_format, *cur_weigth))

        cur_bias = cur_bias.view(-1).tolist()
        bias_format = ''.join(['f' for _ in range(len(cur_bias))])
        file.write(struct.pack(bias_format, *cur_bias))

    # M: grid params
    for grid_elem in model.feature_grid:
        grid_elem = grid_elem.data.reshape(-1).tolist()
        elem_format = ''.join(['f' for _ in range(len(grid_elem))])
        file.write(struct.pack(elem_format, *grid_elem))

    file.flush()
    file.close()


def restore_model(filename):
    file = open(filename, 'rb')

    # M: read header
    n_layers = struct.unpack('B', file.read(1))[0]
    layer_width = struct.unpack('B', file.read(1))[0]
    input_dim = struct.unpack('B', file.read(1))[0]
    output_dim = struct.unpack('B', file.read(1))[0]

    n_grids = struct.unpack('B', file.read(1))[0]
    feature_size = struct.unpack('B', file.read(1))[0]
    grid_dims = struct.unpack(''.join(['I' for _ in range(n_grids)]), file.read(4 * n_grids))
    grid_sizes = []
    for i in range(n_grids):
        sizes = struct.unpack(''.join(['I' for _ in range(grid_dims[i])]), file.read(4 * grid_dims[i]))
        grid_sizes.append(sizes)

    # M: read model params
    net_weights, net_biases, grid_parameters = [], [], []

    def read_in_data(storage: [()]):  # M: give list of target list and number of elements as tuple and read from file
        for tuple_element in storage:
            format_str = ''.join(['f' for _ in range(tuple_element[1])])
            read_data = torch.FloatTensor(struct.unpack(format_str, file.read(4 * tuple_element[1])))
            tuple_element[0].append(read_data)

    read_in_data([(net_weights, input_dim * layer_width), (net_biases, layer_width)])

    for i in range(n_layers-1):
        read_in_data([(net_weights, layer_width * layer_width), (net_biases, layer_width)])

    read_in_data([(net_weights, output_dim * layer_width), (net_biases, output_dim)])

    # M: read grid params
    for i in range(n_grids):
        grid_size = 1
        for elem in grid_sizes[i]:
            grid_size *= elem
        read_in_data([(grid_parameters, feature_size * grid_size)])

    pass
