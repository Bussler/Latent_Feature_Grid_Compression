import torch
from scipy.interpolate import RegularGridInterpolator


# TODO check again if legit
# trilinear interpolation from https://github.com/matthewberger/neurcomp
# e.g. val1 = trilinear_f_interpolation(raw_positions, volume, dataset.min_idx, dataset.max_idx, dataset.vol_res)
def trilinear_f_interpolation(p, f, min_bb, max_bb, res):
    # map points to lattice, and to integer coordinates
    normalized_p = ((p - min_bb.unsqueeze(0)) / ((max_bb - min_bb).unsqueeze(0))) * (res.unsqueeze(0) - 1)
    lattice_p_floor = torch.floor(normalized_p).to(torch.long)
    lattice_p_ceil = torch.ceil(normalized_p).to(torch.long)

    # it's possible that normalized_p has values that are integers? welp, ok, let's account for that...
    min_ref = 1e-12 * torch.ones_like(normalized_p[:1, 0])
    the_diff = torch.max((lattice_p_ceil - lattice_p_floor).to(torch.double), min_ref.to(torch.double))

    # alphas
    alpha = (normalized_p.to(torch.double) - lattice_p_floor.to(torch.double)) / the_diff
    alpha = alpha.to(torch.float)
    one_alpha = 1.0 - alpha

    # x interpolation
    x_interp_y0z0 = one_alpha[:, 0] * f[lattice_p_floor[:, 0], lattice_p_floor[:, 1], lattice_p_floor[:, 2]] + alpha[:,
                                                                                                               0] * f[
                        lattice_p_ceil[:, 0], lattice_p_floor[:, 1], lattice_p_floor[:, 2]]
    x_interp_y1z0 = one_alpha[:, 0] * f[lattice_p_floor[:, 0], lattice_p_ceil[:, 1], lattice_p_floor[:, 2]] + alpha[:,
                                                                                                              0] * f[
                        lattice_p_ceil[:, 0], lattice_p_ceil[:, 1], lattice_p_floor[:, 2]]
    x_interp_y0z1 = one_alpha[:, 0] * f[lattice_p_floor[:, 0], lattice_p_floor[:, 1], lattice_p_ceil[:, 2]] + alpha[:,
                                                                                                              0] * f[
                        lattice_p_ceil[:, 0], lattice_p_floor[:, 1], lattice_p_ceil[:, 2]]
    x_interp_y1z1 = one_alpha[:, 0] * f[lattice_p_floor[:, 0], lattice_p_ceil[:, 1], lattice_p_ceil[:, 2]] + alpha[:,
                                                                                                             0] * f[
                        lattice_p_ceil[:, 0], lattice_p_ceil[:, 1], lattice_p_ceil[:, 2]]

    # y interpolation
    y_interp_z0 = one_alpha[:, 1] * x_interp_y0z0 + alpha[:, 1] * x_interp_y1z0
    y_interp_z1 = one_alpha[:, 1] * x_interp_y0z1 + alpha[:, 1] * x_interp_y1z1

    # final interpolated value
    interp_val = one_alpha[:, 2] * y_interp_z0 + alpha[:, 2] * y_interp_z1

    return interp_val


def finite_difference_trilinear_grad(p,f,min_bb,max_bb,res,scale=None):
    x_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)
    y_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)
    z_step = ((max_bb-min_bb)/(res-1)).unsqueeze(0)

    x_step[:,1:] = 0
    y_step[:,0] = 0
    y_step[:,2] = 0
    z_step[:,:2] = 0

    x_negative = p-x_step
    x_positive = p+x_step
    y_negative = p-y_step
    y_positive = p+y_step
    z_negative = p-z_step
    z_positive = p+z_step

    x_negative[x_negative[:,0] < min_bb[0],0] = min_bb[0]
    y_negative[y_negative[:,1] < min_bb[1],1] = min_bb[1]
    z_negative[z_negative[:,2] < min_bb[2],2] = min_bb[2]
    x_positive[x_positive[:,0] > max_bb[0],0] = max_bb[0]
    y_positive[y_positive[:,1] > max_bb[1],1] = max_bb[1]
    z_positive[z_positive[:,2] > max_bb[2],2] = max_bb[2]

    if scale is None:
        x_diff = 2*(x_positive[:,0]-x_negative[:,0]) / (max_bb[0]-min_bb[0])
        y_diff = 2*(y_positive[:,1]-y_negative[:,1]) / (max_bb[1]-min_bb[1])
        z_diff = 2*(z_positive[:,2]-z_negative[:,2]) / (max_bb[2]-min_bb[2])
    else:
        x_diff = 2*scale[0]*(x_positive[:,0]-x_negative[:,0]) / (max_bb[0]-min_bb[0])
        y_diff = 2*scale[1]*(y_positive[:,1]-y_negative[:,1]) / (max_bb[1]-min_bb[1])
        z_diff = 2*scale[2]*(z_positive[:,2]-z_negative[:,2]) / (max_bb[2]-min_bb[2])
    #

    x_deriv = (trilinear_f_interpolation(x_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(x_negative,f,min_bb,max_bb,res))/x_diff
    y_deriv = (trilinear_f_interpolation(y_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(y_negative,f,min_bb,max_bb,res))/y_diff
    z_deriv = (trilinear_f_interpolation(z_positive,f,min_bb,max_bb,res) - trilinear_f_interpolation(z_negative,f,min_bb,max_bb,res))/z_diff

    return torch.cat((x_deriv.unsqueeze(1),y_deriv.unsqueeze(1),z_deriv.unsqueeze(1)),1)


# generate RegularGridInterpolator for trilinear interpolation
# volume_interpolator = generate_RegularGridInterpolator(volume)  # M: used for accessing the volume with indices
# ground_truth_volume = volume_interpolator(raw_positions.cpu())
# tg2 = numpy.gradient(volume.numpy())
# test = tg2(raw_positions[0,0],raw_positions[0,1],raw_positions[0,2])
def generate_RegularGridInterpolator(volume):
    x = torch.linspace(0, volume.shape[0] - 1, volume.shape[0])
    y = torch.linspace(0, volume.shape[1] - 1, volume.shape[1])
    z = torch.linspace(0, volume.shape[2] - 1, volume.shape[2])
    fn = RegularGridInterpolator((x, y, z), volume.numpy()) # M: TODO: Option to pass Tensor here?

    return fn
