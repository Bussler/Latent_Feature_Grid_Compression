import torch as th
import numpy as np
from pyevtk.hl import imageToVTK


# taken from https://github.com/matthewberger/neurcomp
def field_from_net(dataset, net, is_cuda, tiled_res=32, verbose=False):
    target_res = dataset.vol_res_touple
    #target_res = dataset.vol_res
    full_vol = th.zeros(target_res)
    for xdx in np.arange(0,target_res[0],tiled_res):
        if verbose:
            print('x',xdx,'/',target_res[0])
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
        for ydx in np.arange(0,target_res[1],tiled_res):
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            for zdx in np.arange(0,target_res[2],tiled_res):
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]

                tile_resolution = th.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=th.int)

                min_alpha_bb = th.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=th.float)
                max_alpha_bb = th.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=th.float)
                min_bounds = dataset.min_idx + min_alpha_bb*(dataset.max_idx-dataset.min_idx)
                max_bounds = dataset.min_idx + max_alpha_bb*(dataset.max_idx-dataset.min_idx)
                #min_bounds = dataset.min_bb + min_alpha_bb * (dataset.max_bb - dataset.min_bb)
                #max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)

                with th.no_grad():
                    start = min_bounds / (dataset.max_idx - dataset.min_idx)
                    end = max_bounds / (dataset.max_idx - dataset.min_idx)
                    norm_indices = dataset.generate_indices(start,end,tile_resolution)
                    norm_indices = 2.0 * norm_indices - 1.0
                    tile_positions = dataset.scales.view(1,1,1,3)*norm_indices
                    #tile_positions = dataset.scales.view(1, 1, 1, 3) * dataset.tile_sampling(min_bounds, max_bounds, tile_resolution)
                    if is_cuda:
                        tile_positions = tile_positions.unsqueeze(0).cuda()
                    tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)
                    full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.cpu()
                #
            #
        #
    #
    return full_vol
#


# M: Print and return error statistics between a prediction and a ground truth volume.
# M: Both parameters should be torch.Tensor of the same size.
def calculate_deviation_statistics(prediction, ground_truth):
    diff_vol = ground_truth - prediction
    sqd_max_diff = (th.max(ground_truth) - th.min(ground_truth)) ** 2  # M: max für tthresh anpassen!!
    l1_diff = th.mean(th.abs(diff_vol))
    mse = th.mean(th.pow(diff_vol, 2.0))
    psnr = 10 * th.log10(sqd_max_diff / mse)
    print('PSNR:', psnr, 'l1:', l1_diff, 'mse:', mse, 'rmse:', th.sqrt(mse))
    return psnr.item(), l1_diff.item(), mse.item(), th.sqrt(mse).item()


# taken from https://github.com/matthewberger/neurcomp
def tiled_net_out(dataset, net, is_cuda, gt_vol=None, evaluate=True, write_vols=False, filename='vol'):
    if is_cuda:
        net = net.cuda()
    net.eval()
    full_vol = field_from_net(dataset, net, is_cuda, tiled_res=32)
    psnr = 0
    print('writing to VTK...')
    if evaluate and gt_vol is not None:
        psnr, l1_diff, mse, rmse = calculate_deviation_statistics(full_vol, gt_vol)

    if write_vols:
        imageToVTK(filename, pointData = {'sf':full_vol.numpy()})
        if gt_vol is not None:
            imageToVTK('gt', pointData = {'sf':gt_vol.numpy()})
    #

    print('back to training...')
    net.train()
    return psnr if not evaluate else psnr, l1_diff, mse, rmse