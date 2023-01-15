import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding
from data.IndexDataset import get_tensor, IndexDataset
import training.learning_rate_decay as lrdecay
from data.Interpolation import trilinear_f_interpolation
from model.Smallify_Dropout import SmallifyDropout, SmallifyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = None


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

            # M: TODO Debugging

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

    model = solve_model(model, optimizer, lrStrategy, loss_criterion, drop_loss, volume,
                           dataset, data_loader, args, verbose)
