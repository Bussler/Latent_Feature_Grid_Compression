import numpy as np
import torch
from model.model_utils import setup_model, restore_model
from visualization.pltUtils import dict_from_file
from visualization.OutputToVTK import tiled_net_out
from data.IndexDataset import get_tensor, IndexDataset


def infer_model(model, dataset, volume):
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=True)
    pass


def create_model_from_checkpoint(args):
    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], '', args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])

    return model


def create_model_from_binary_file(args):
    model = restore_model(args['binary_checkpoint_path'])
    return model


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help='path to config of model; is required')

    parser.add_argument("--reconstruct", type=str, required=True,
                        help='from where to reconstruct the model from: binary: from binary files;'
                             ' checkpoint: from model checkpoint')

    args = vars(parser.parse_args())

    config = dict_from_file(args['config_path'])

    volume = get_tensor(config['data'])
    dataset = IndexDataset(volume, config['sample_size'])

    if args['reconstruct'] == 'checkpoint':
        model = create_model_from_checkpoint(config)
    if args['reconstruct'] == 'binary':
        model = create_model_from_binary_file(config)
    if args['reconstruct'] != 'binary' and args['reconstruct'] != 'checkpoint':
        print("Wrong --reconstruct parameter! Specify either 'binary' or 'checkpoint'")

    infer_model(model, dataset, volume)
