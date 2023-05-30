import os
import torch
import numpy as np
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from model.Smallify_Dropout import SmallifyDropout
from model.Straight_Through_Dropout import MaskedWavelet_Straight_Through_Dropout, Straight_Through_Dropout
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.Variational_Dropout_Layer import VariationalDropout
import struct
import re
from sklearn.cluster import KMeans
import math
from typing import List, Tuple


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


def kmeans_quantization(w,q):
        weight_feat = w
        kmeans = KMeans(n_clusters=q,n_init=4).fit(weight_feat)
        return kmeans.labels_.tolist(),kmeans.cluster_centers_.reshape(q).tolist()
    
    
def ints_to_bits_to_bytes(all_ints,n_bits):
        f_str = '#0'+str(n_bits+2)+'b'
        bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
        n_bytes = len(bit_string)//8
        the_leftover = len(bit_string)%8>0
        if the_leftover:
            n_bytes+=1
        the_bytes = bytearray()
        for b in range(n_bytes):
            bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
            the_bytes.append(int(bin_val,2))
        return the_bytes,the_leftover


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
    zeroes = []
    for grid in model.feature_grid:
        grid_sizes.append(torch.count_nonzero(grid))
        zeroes.append(torch.numel(grid) - torch.count_nonzero(grid))
        
    bit_precision = 8 # TODO: extract this into user-variable
    n_clusters = int(math.pow(2, bit_precision))

    # M: header
    file.write(struct.pack('B', n_layers))
    file.write(struct.pack('B', layer_width))
    file.write(struct.pack('B', input_dim))
    file.write(struct.pack('B', input_channel))
    file.write(struct.pack('B', output_dim))
    file.write(struct.pack('B', bit_precision))

    file.write(struct.pack('B', grid_size))

    file.write(struct.pack('B', n_grids))
    file.write(struct.pack('B', feature_size))
    for grid in grid_sizes:
        file.write(struct.pack('I', grid))
    for zero in zeroes:
        file.write(struct.pack('I', zero))

    # M: model params
    net_weights, net_biases = get_net_weights_biases(model)
    
    first_weight, first_bias = net_weights[0].view(-1).tolist(), net_biases[0].view(-1).tolist()
    weight_format = ''.join(['f' for _ in range(len(first_weight))])
    file.write(struct.pack(weight_format, *first_weight))
    bias_format = ''.join(['f' for _ in range(len(first_bias))])
    file.write(struct.pack(bias_format, *first_bias))
    
    def write_tensor_quantized(cur_weigth):
        cur_weigth = cur_weigth.view(-1).unsqueeze(1).numpy()
        labels, centers = kmeans_quantization(cur_weigth, n_clusters)
        w = centers
        w_format = ''.join(['f' for _ in range(len(w))])
        file.write(struct.pack(w_format, *w))
        weight_bin, is_leftover = ints_to_bits_to_bytes(labels, bit_precision)
        file.write(weight_bin)

        # encode non-pow-2 as 16-bit integer
        if bit_precision % 8 != 0:
            file.write(struct.pack('I', labels[-1]))
        
    
    for cur_weigth, cur_bias in zip(net_weights[1:-1], net_biases[1:-1]):   
             
        # M: quantize weights prior to writing
        write_tensor_quantized(cur_weigth)

        # M: bias
        cur_bias = cur_bias.view(-1).tolist()
        bias_format = ''.join(['f' for _ in range(len(cur_bias))])
        file.write(struct.pack(bias_format, *cur_bias))
    
    last_weight, last_bias = net_weights[-1].view(-1).tolist(), net_biases[-1].view(-1).tolist()
    weight_format = ''.join(['f' for _ in range(len(last_weight))])
    file.write(struct.pack(weight_format, *last_weight))
    bias_format = ''.join(['f' for _ in range(len(last_bias))])
    file.write(struct.pack(bias_format, *last_bias))

    # M: grid params
    mask_string = ""

    for grid_elem in model.feature_grid:
        grid_elem = grid_elem.data.reshape(-1)

        # M: generate mask
        for g_elem in grid_elem:
            mask_string = mask_string + "1" if g_elem != 0.0 else mask_string + "0"

        # M: delete 0 elements from grid
        nonzero_indices = torch.nonzero(grid_elem)
        grid_elem = grid_elem[nonzero_indices].squeeze()
        # M: quantize grid elements prior to writing
        write_tensor_quantized(grid_elem)

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
    
    bit_precision = struct.unpack('B', file.read(1))[0]
    n_clusters = int(math.pow(2, bit_precision))

    grid_size = struct.unpack('B', file.read(1))[0]

    n_grids = struct.unpack('B', file.read(1))[0]
    feature_size = struct.unpack('B', file.read(1))[0]
    grid_sizes = []
    zeros = []
    for i in range(n_grids):
        size = struct.unpack('I', file.read(4))[0]
        grid_sizes.append(size)
    for i in range(n_grids):
        zero = struct.unpack('I', file.read(4))[0]
        zeros.append(zero)

    net_weights, net_biases, grid_parameters = [], [], []

    # M: give list of target list and number of elements as tuple and read from file
    def read_in_data(storage: List[Tuple], convert_to_Tensor=True):
        for tuple_element in storage:
            format_str = ''.join(['f' for _ in range(tuple_element[1])])
            if convert_to_Tensor:
                read_data = torch.FloatTensor(struct.unpack(format_str, file.read(4 * tuple_element[1])))
            else:
                read_data = np.array(struct.unpack(format_str, file.read(4 * tuple_element[1])))
            tuple_element[0].append(read_data)
            
    def read_in_data_quantized(storage: List, n_weights: int, convert_to_Tensor=True):
        weight_size = (n_weights*bit_precision)//8
        if (n_weights*bit_precision)%8 != 0:
            weight_size+=1
        c_format = ''.join(['f' for _ in range(n_clusters)])
        centers = torch.FloatTensor(struct.unpack(c_format, file.read(4*n_clusters)))
        inds = file.read(weight_size)
        bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
        w_inds = torch.LongTensor([int(bits[bit_precision*i:bit_precision*i+bit_precision],2) for i in range(n_weights)])

        if bit_precision%8 != 0:
            next_bytes = file.read(4)
            w_inds[-1] = struct.unpack('I', next_bytes)[0]
            
        w_quant = centers[w_inds]
        
        if not convert_to_Tensor:
            w_quant = w_quant.numpy()        
        
        storage.append(w_quant)

    # M: read model params
    read_in_data([(net_weights, input_dim * layer_width), (net_biases, layer_width)])

    for i in range(n_layers-1):
        read_in_data_quantized(net_weights, layer_width * layer_width)
        # M: read in unquantized bias
        read_in_data([(net_biases, layer_width)])

    read_in_data([(net_weights, output_dim * layer_width), (net_biases, output_dim)])

    # M: read grid params
    all_elements = sum(grid_sizes) + sum(zeros)
    mask_reconstructed = read_binary(filename + "_mask.bnr", all_elements)

    for length in grid_sizes:
        read_in_data_quantized(grid_parameters, length, convert_to_Tensor=False)

    # M: insert pruned 0 into grid Tensors:
    tensor_grid_params = []
    mask_pointer = 0
    for cur_size, cur_zeros, cur_array in zip(grid_sizes, zeros, grid_parameters):
        for i in range(cur_size+cur_zeros):
            if mask_reconstructed[mask_pointer] == '0':
                cur_array = np.insert(cur_array, i, 0.0)
            mask_pointer += 1
        tensor_grid_params.append(torch.tensor(cur_array, dtype=torch.float32))


    # M: reconstruct model
    model = setup_model(input_channel = input_channel, hidden_channel = layer_width, out_channel = output_dim,
                        num_layer = n_layers, embedding_type = "fourier", n_embedding_freq = 2, drop_type = "",
                        drop_momentum = 0.025, drop_threshold = 0.75, wavelet_filter = "db2",
                        grid_features = feature_size, grid_size = grid_size, checkpoint_path = "")

    wdx, bdx, gdx = 0, 0, 0
    for name, parameters in model.named_parameters():
        if re.match(r'.*grid.*', name, re.I):
            w_shape = parameters.data.shape
            parameters.data = tensor_grid_params[gdx].view(w_shape)
            gdx += 1
        if re.match(r'.*.weight', name, re.I):
            w_shape = parameters.data.shape
            parameters.data = net_weights[wdx].view(w_shape)
            wdx += 1
        if re.match(r'.*.bias', name, re.I):
            b_shape = parameters.data.shape
            parameters.data = net_biases[bdx].view(b_shape)
            bdx += 1

    file.close()

    return model
