import numpy as np
import torch
import pywt
import pywt.data
import matplotlib.pyplot as plt
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from model.model_utils import write_dict, setup_model
from visualization.pltUtils import dict_from_file
from visualization.OutputToVTK import tiled_net_out
from data.IndexDataset import get_tensor, IndexDataset


def get_FeatureGrid_NWWeight(state_dict):

    featuregrid = {}
    weights_biases = {}

    for key, value in state_dict.items():
        if 'feature_grid' in key:
            featuregrid[key] = value
        if 'net_layers' in key:
            weights_biases[key] = value
    return featuregrid, weights_biases


def calculate_quantized_state_dict(state_dict):

    features, nw_weights = get_FeatureGrid_NWWeight(state_dict)

    for key, value in features.items():
        minv = torch.min(value)
        maxv = torch.max(value)
        scale, zero_point = 0.02, 0 #0.02 0.08
        dtype = torch.qint8
        quant_tensor = torch.quantize_per_tensor(value.data, scale, zero_point, dtype)
        dequant_tensor = quant_tensor.dequantize()
        state_dict[key] = dequant_tensor

    for key, value in nw_weights.items():
        scale, zero_point = 1e-4, 0
        dtype = torch.qint32
        quant_tensor = torch.quantize_per_tensor(value.data, scale, zero_point, dtype)
        dequant_tensor = quant_tensor.dequantize()
        state_dict[key] = dequant_tensor

    return state_dict



def calculate_Quantized_Qualita_Compression(checkpoint_path, config_path):


    args = dict_from_file(config_path)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], args['drop_type'], args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])

    model.load_state_dict(torch.load(checkpoint_path))

    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    num_net_feature_params = 0
    num_net_params = 0
    numzeroes = 53034.0

    for name, layer in model.named_parameters():
        if 'feature_grid' in name:
            num_net_feature_params += layer.numel()
        else:
            if 'drop' not in name:
                num_net_params += layer.numel()

    num_net_feature_params -= numzeroes
    all_params = num_net_feature_params + num_net_params

    all_params_quantized = (num_net_feature_params/4) + (num_net_params/2)

    compression_ratio = dataset.n_voxels / (all_params)
    compression_ratio_quantized = dataset.n_voxels / (all_params_quantized)

    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=False)
    print('PSNR: ', psnr, ' compression ratio:', compression_ratio)


    state_dict = torch.load(checkpoint_path)
    quantized_state_dict = calculate_quantized_state_dict(state_dict)
    model.load_state_dict(quantized_state_dict)

    psnr_quant, l1_diff_quant, mse_quant, rmse_quant = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=False)
    print('PSNR_Quant: ', psnr_quant, ' Ratio: ', psnr_quant/psnr)
    print('Compr_Quant: ', compression_ratio_quantized, ' Ratio: ', compression_ratio_quantized / compression_ratio)

    pass



if __name__ == '__main__':
    checkpoint_path = '../experiments/Tests/Quantization/Smallify/mhd_p_/model.pth'
    config_path = '../experiments/Tests/Quantization/Smallify/mhd_p_/config.txt'

    calculate_Quantized_Qualita_Compression(checkpoint_path, config_path)
