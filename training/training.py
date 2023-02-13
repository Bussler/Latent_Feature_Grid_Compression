import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from data.IndexDataset import get_tensor, IndexDataset
import training.learning_rate_decay as lrdecay
from data.Interpolation import trilinear_f_interpolation
from model.Smallify_Dropout import SmallifyDropout, SmallifyLoss
from model.Variational_Dropout_Layer import VariationalDropoutLoss, Variance_Model, VariationalDropout
from visualization.OutputToVTK import tiled_net_out
from model.model_utils import write_dict, setup_model
from wavelet_transform.Torch_Wavelet_Transform import WaveletFilter3d
from copy import deepcopy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = None


def evaluate_model_training(model, dataset, volume, zeros, args, verbose=True):
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=False)

    info = {}
    num_net_params = 0

    for name, layer in model.named_parameters():
        if 'drop' not in name:
            num_net_params += layer.numel()
    compression_ratio = dataset.n_voxels / (num_net_params-zeros.item())
    compr_rmse = compression_ratio / rmse

    if verbose:
        print("Trained Model: ", num_net_params, " parameters; ", zeros.item(), 'of them Zero; ',
              compression_ratio, " compression ratio")
        print("Model: \n", model)

    info['volume_size'] = dataset.vol_res.tolist()
    info['volume_num_voxels'] = dataset.n_voxels
    info['num_parameters'] = num_net_params
    info['num_zeros'] = zeros.item()
    info['compression_ratio'] = compression_ratio
    info['psnr'] = psnr
    info['l1_diff'] = l1_diff
    info['mse'] = mse
    info['rmse'] = rmse
    info['compr_rmse'] = compr_rmse

    writer.add_scalar("compression_ratio", compression_ratio)
    writer.add_scalar("zeroes", zeros.item())
    writer.add_scalar("psnr", psnr)
    writer.add_scalar("mse", mse)
    writer.add_scalar("rmse", rmse)
    writer.add_scalar("compr_rmse", compr_rmse)

    # M: Safe more data

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ExperimentPath, 'model.pth'))
    args['checkpoint_path'] = os.path.join(ExperimentPath, 'model.pth')

    write_dict(info, 'info.txt', ExperimentPath)
    write_dict(args, 'config.txt', ExperimentPath)

    return info


def solve_model(model_init, optimizer, lr_strategy, loss_criterion, drop_loss,
                volume, dataset, data_loader, args, verbose=True):

    model = model_init
    voxel_seen = 0.0
    volume_passes = 0.0
    step_iter = 0
    lr_decay_stop = False

    #if args['drop_type'] and args['drop_type'] == 'variational':
    #    variance_model = Variance_Model()
    #    variance_model.to(device)
    #    variance_model.train()
    #    optimizer.add_param_group({'params': variance_model.parameters()})

    # M: Training Loop
    while int(volume_passes) + 1 < args['max_pass'] and not lr_decay_stop:  # M: epochs

        for idx, data in enumerate(data_loader):
            step_iter += 1

            # M: Access data
            raw_positions, norm_positions = data

            raw_positions = raw_positions.to(device)  # M: Tensor of size [batch_size, sample_size, 3]
            norm_positions = norm_positions.to(device)
            raw_positions = raw_positions.view(-1, args['d_in'])  # M: Tensor of size [batch_size x sample_size, 3]
            norm_positions = norm_positions.view(-1, args['d_in'])
            norm_positions.requires_grad = True  # M: For gradient calculation of nw

            # M: NW prediction
            optimizer.zero_grad()
            predicted_volume = model(norm_positions)
            predicted_volume = predicted_volume.squeeze(-1)  # M: Tensor of size [batch_size x dataset.sample_size, 1]

            # M: Calculate loss
            ground_truth_volume = trilinear_f_interpolation(raw_positions, volume,
                                                            dataset.min_idx.to(device), dataset.max_idx.to(device),
                                                            dataset.vol_res.to(device))


            # M: Used for Learning rate decay
            prior_volume_passes = int(voxel_seen / dataset.n_voxels)
            voxel_seen += ground_truth_volume.shape[0]
            volume_passes = voxel_seen / dataset.n_voxels

            # M: Loss calculation
            if args['drop_type'] == 'variational':
                #variational_variance = variance_model(norm_positions)
                #variational_variance = variational_variance.squeeze(-1)

                variational_variance = torch.ones_like(predicted_volume).fill_(args['variational_sigma'])  # -7.0, 5e-04

                complete_loss, Log_Likelyhood, mse, Dkl_sum, weight_sum = drop_loss(model, predicted_volume,
                                                                                    ground_truth_volume,
                                                                                    variational_variance,
                                                                                    args['weight_dkl_multiplier'])
            else:
                vol_loss = loss_criterion(predicted_volume, ground_truth_volume)
                if drop_loss is not None:
                    d_loss = drop_loss(model)
                else:
                    d_loss = torch.zeros_like(vol_loss)
                complete_loss = vol_loss + d_loss

            complete_loss.backward()
            optimizer.step()

            # M: Update lr according to strategy
            if lr_strategy.decay_learning_rate(prior_volume_passes, volume_passes, complete_loss):
                lr_decay_stop = True
                break

            # M: Debugging
            if args['drop_type'] == 'variational':
                writer.add_scalar("loss", complete_loss, step_iter)
                writer.add_scalar("volume_loss", mse, step_iter)
                writer.add_scalar("Log_Likelyhood_loss", Log_Likelyhood, step_iter)
                writer.add_scalar("DKL_loss", Dkl_sum, step_iter)
                writer.add_scalar("Weight_loss", weight_sum, step_iter)
            else:
                writer.add_scalar("loss", complete_loss.item(), step_iter)
                writer.add_scalar("volume_loss", vol_loss.item(), step_iter)
                writer.add_scalar("drop_loss", d_loss.item(), step_iter)

            # M: Print training statistics:
            if idx % 100 == 0 and verbose:
                if args['drop_type'] == 'variational':
                    print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, LL: {:.4f}, DKL: {:.4f}, complete_loss: {:.4f}'.
                          format(volume_passes, args['max_pass'], mse, Log_Likelyhood, Dkl_sum, complete_loss))

                    #valid_fraction = []
                    #droprates = []
                    #for module in model.drop.modules():
                    #    if isinstance(module, VariationalDropout):
                    #        d, dropr = module.get_valid_fraction()
                    #        valid_fraction.append(d)
                    #        droprates.append(dropr)
                    #writer.add_histogram("droprates_layer1", droprates[0], step_iter)
                    #writer.add_histogram("droprates_layer2", droprates[1], step_iter)
                    #writer.add_histogram("droprates_layer3", droprates[2], step_iter)
                    #writer.add_histogram("droprates_layer4", droprates[3], step_iter)
                    #print('Valid Fraction: ', valid_fraction)
                else:
                    print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, drop_loss: {:.4f}, complete_loss: {:.4f}'.
                          format(volume_passes, args['max_pass'], vol_loss.item(), d_loss.item(), complete_loss.item()))

            if (int(volume_passes)) >= args['max_pass']:
                break

    return model


def training(args, verbose=True):
    # M: Get volume data, set up data
    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                             num_workers=args['num_workers'])  # M: create dataloader from dataset to use in training
    volume = volume.to(device)

    model = setup_model(args['d_in'], args['n_hidden_size'], args['d_out'], args['n_layers'], args['embedding_type'],
                        args['n_embedding_freq'], args['drop_type'], args['drop_momentum'], args['drop_threshold'],
                        args['wavelet_filter'], args['grid_features'], args['grid_size'], args['checkpoint_path'])
    model.to(device)
    model.train()

    # M: Setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    lrStrategy = lrdecay.LearningRateDecayStrategy.create_instance(args, optimizer)
    loss_criterion = torch.nn.MSELoss().to(device)
    if args['drop_type'] == 'variational':
        drop_loss = VariationalDropoutLoss(size_volume=dataset.n_voxels,
                                           batch_size=args['batch_size']*args['sample_size'],
                                           weight_dkl=args['lambda_drop_loss'],
                                           weight_weights=args['lambda_weight_loss'])
    else:
        drop_loss = SmallifyLoss(weight_l1=args['lambda_drop_loss'], weight_l2=args['lambda_weight_loss'])

    # M: Setup Tensorboard writer
    global writer
    if args['Tensorboard_log_dir']:
        writer = SummaryWriter(args['Tensorboard_log_dir'])
        write_dict(args, 'config.txt', args['Tensorboard_log_dir'])
    else:
        writer = SummaryWriter('runs/'+args['expname'])

    # M: Training and finetuning
    args_first = deepcopy(args)
    args_first['max_pass'] *= (2.0 / 3.0)

    model = solve_model(model, optimizer, lrStrategy, loss_criterion, drop_loss, volume,
                        dataset, data_loader, args_first, verbose)

    zeros = model.save_dropvalues_on_grid(device)

    # M: Finetuning
    args_second = deepcopy(args)
    args_second['max_pass'] *= (1.0 / 3.0)
    args_second['drop_type'] = ''
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'] / 100.0)

    model = solve_model(model, optimizer, lrStrategy, loss_criterion, None, volume,
                        dataset, data_loader, args_second, verbose)

    info = evaluate_model_training(model, dataset, volume, zeros, args, verbose)
    writer.close()
    return info
