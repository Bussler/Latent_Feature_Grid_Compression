import os
import torch
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from model.Smallify_Dropout import SmallifyDropout
from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout, Straight_Through_Dropout
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.Variational_Dropout_Layer import VariationalDropout
import struct
import re
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


def binary_writing(mask_string, filename):
    n_bytes = len(mask_string) // 8
    the_bytes = bytearray()
    for b in range(n_bytes):
        bin_val = mask_string[8*b:8*b+8]
        the_bytes.append(int(bin_val,2))

    with open(filename, "wb") as file:
        file.write(the_bytes)

        if len(mask_string) % 8 != 0:
            bin_val = mask_string[(8*n_bytes):]
            bin_val = bin_val + '0' * (8 - len(bin_val))
            barr = bytearray([int(bin_val, 2)])
            file.write(barr)


def read_binary(filename, num_bits):
    with open(filename, "rb") as file:
        b_size = (num_bits // 8) + 1 if num_bits % 8 != 0 else num_bits // 8

        inds = file.read(b_size)
        bits = ''.join(format(byte, '0' + str(8) + 'b') for byte in inds)
        file.close()
        return bits


def store_model_parameters(model, filename):
    # M: basic information
    file = open(filename, 'wb')

    n_layers = model.num_layer
    layer_width = model.hidden_width
    input_dim = model.input_channel
    input_channel = model.d_in
    output_dim = model.output_channel

    grid_size = model.shape_array[-1][0]

    n_grids = len(model.feature_grid)
    feature_size = model.feature_grid[0].shape[0]
    grid_sizes = []
    for grid in model.feature_grid:
        grid_sizes.append(torch.numel(grid))

    # M: header
    file.write(struct.pack('B', n_layers))
    file.write(struct.pack('B', layer_width))
    file.write(struct.pack('B', input_dim))
    file.write(struct.pack('B', input_channel))
    file.write(struct.pack('B', output_dim))

    file.write(struct.pack('B', grid_size))

    file.write(struct.pack('B', n_grids))
    file.write(struct.pack('B', feature_size))
    for grid in grid_sizes:
        file.write(struct.pack('I', grid))

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
    mask_string = ""

    for grid_elem in model.feature_grid:
        # M: TODO: delete 0 elements from grid

        grid_elem = grid_elem.data.reshape(-1).tolist()
        elem_format = ''.join(['f' for _ in range(len(grid_elem))])
        file.write(struct.pack(elem_format, *grid_elem))

        for g_elem in grid_elem:
            mask_string = mask_string + "1" if g_elem != 0.0 else mask_string + "0"

    binary_writing(mask_string, filename+"_mask.bnr")

    file.flush()
    file.close()


def restore_model(filename):
    file = open(filename, 'rb')

    # M: read header
    n_layers = struct.unpack('B', file.read(1))[0]
    layer_width = struct.unpack('B', file.read(1))[0]
    input_dim = struct.unpack('B', file.read(1))[0]
    input_channel = struct.unpack('B', file.read(1))[0]
    output_dim = struct.unpack('B', file.read(1))[0]

    grid_size = struct.unpack('B', file.read(1))[0]

    n_grids = struct.unpack('B', file.read(1))[0]
    feature_size = struct.unpack('B', file.read(1))[0]
    grid_sizes = []
    for i in range(n_grids):
        size = struct.unpack('I', file.read(4))[0]
        grid_sizes.append(size)

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
    all_elements = sum(grid_sizes)
    mask_reconstructed = read_binary(filename + "_mask.bnr", all_elements)

    for length in grid_sizes:
        read_in_data([(grid_parameters, length)])

    # M: TODO insert pruned 0 into grid Tensors:


    # M: reconstruct model
    model = setup_model(input_channel = input_channel, hidden_channel = layer_width, out_channel = output_dim,
                        num_layer = n_layers, embedding_type = "fourier", n_embedding_freq = 2, drop_type = "",
                        drop_momentum = 0.025, drop_threshold = 0.75, wavelet_filter = "db2",
                        grid_features = feature_size, grid_size = grid_size, checkpoint_path = "")

    wdx, bdx, gdx = 0, 0, 0
    for name, parameters in model.named_parameters():
        if re.match(r'.*grid.*', name, re.I):
            w_shape = parameters.data.shape
            parameters.data = grid_parameters[gdx].view(w_shape)
            gdx += 1
        if re.match(r'.*.weight', name, re.I):
            w_shape = parameters.data.shape
            parameters.data = net_weights[wdx].view(w_shape)
            wdx += 1
        if re.match(r'.*.bias', name, re.I):
            b_shape = parameters.data.shape
            parameters.data = net_biases[bdx].view(b_shape)
            bdx += 1

    return model
