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
from visualization.OutputToVTK import tiled_net_out
from model.model_utils import write_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = None


def evaluate_model_training(model, dataset, volume, args, verbose=True):
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=False)

    info = {}
    num_net_params = 0
    for layer in model.parameters():
        num_net_params += layer.numel()
    compression_ratio = dataset.n_voxels / num_net_params
    compr_rmse = compression_ratio / rmse

    if verbose:
        print("Trained Model: ", num_net_params, " parameters; ", compression_ratio, " compression ratio")

    info['volume_size'] = dataset.vol_res.tolist()
    info['volume_num_voxels'] = dataset.n_voxels
    info['num_parameters'] = num_net_params
    info['compression_ratio'] = compression_ratio
    info['psnr'] = psnr
    info['l1_diff'] = l1_diff
    info['mse'] = mse
    info['rmse'] = rmse
    info['compr_rmse'] = compr_rmse

    # M: Safe more data

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)

    #torch.save(model.state_dict(), os.path.join(ExperimentPath, 'model.pth'))
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
            vol_loss = loss_criterion(predicted_volume, ground_truth_volume)
            d_loss = drop_loss(model)
            complete_loss = vol_loss + d_loss

            complete_loss.backward()
            optimizer.step()

            # M: Update lr according to strategy
            if lr_strategy.decay_learning_rate(prior_volume_passes, volume_passes, complete_loss):
                lr_decay_stop = True
                break

            # M: Debugging
            writer.add_scalar("loss", complete_loss.item(), step_iter)
            writer.add_scalar("volume_loss", vol_loss.item(), step_iter)
            writer.add_scalar("drop_loss", d_loss.item(), step_iter)

            # M: Print training statistics:
            if idx % 100 == 0 and verbose:
                print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, drop_loss: {:.4f}, complete_loss: {:.4f}'.format(
                    volume_passes, args['max_pass'], vol_loss.item(), d_loss.item(), complete_loss.item()))

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

    # M: Setup Latent_Feature_Grid
    size_tensor = (16, 32, 32, 32)
    feature_grid = torch.empty(size_tensor).uniform_(0, 1)

    # M: Setup Drop-Layer for grid
    drop_layer = SmallifyDropout(feature_grid.shape[1:])

    # M: Setup Embedder
    embedder = FourierEmbedding(n_freqs=2, input_dim=3)

    # M: Setup model
    model = Feature_Grid_Model(embedder, feature_grid, drop_layer)
    model.to(device)
    model.train()

    # M: Setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    lrStrategy = lrdecay.LearningRateDecayStrategy.create_instance(args, optimizer)
    loss_criterion = torch.nn.MSELoss().to(device)
    drop_loss = SmallifyLoss(weight_l1=1.e-6, weight_l2=0.)

    # M: Setup Tensorboard writer
    global writer
    if args['Tensorboard_log_dir']:
        writer = SummaryWriter(args['Tensorboard_log_dir'])
    else:
        writer = SummaryWriter('runs/'+args['expname'])

    model = solve_model(model, optimizer, lrStrategy, loss_criterion, drop_loss, volume,
                           dataset, data_loader, args, verbose)

    model.save_dropvalues_on_grid(device)

    info = evaluate_model_training(model, dataset, volume, args, verbose)
    writer.close()
    return info
