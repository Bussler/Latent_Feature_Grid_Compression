import numpy as np
import torch
from torch.utils.data.dataset import Dataset


# M: norm volume to [minN, maxN]
def normalize_volume(volume, minV, maxV, minN, maxN):
    return (maxN-minN) * ((volume - minV) / (maxV - minV)) + minN


def get_tensor_from_numpy(filepath):
    np_volume = np.load(filepath).astype(np.float32)
    volume = torch.from_numpy(np_volume)

    minV = torch.min(volume)
    maxV = torch.max(volume)
    volume = normalize_volume(volume, minV, maxV, -1.0, 1.0)

    print('Loaded Numpy Volume Successfully. Shape of: ', volume.shape)
    return volume


def get_tensor_from_hdf5(filepath):
    import h5py
    with h5py.File(filepath, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]  # M: numpy array
        ds_arr = np.squeeze(ds_arr)
        volume = torch.from_numpy(ds_arr)

        minV = torch.min(volume)
        maxV = torch.max(volume)
        volume = normalize_volume(volume, minV, maxV, -1.0, 1.0)

        print('Loaded HDF5 Volume Successfully. Shape of: ', volume.shape)
        return volume


def get_tensor(filepath):

    if filepath.endswith('.npy'):
        return get_tensor_from_numpy(filepath)
    if filepath.endswith('.cvol'):
        from data.pyrendererSupport import get_tensor_from_cvol
        return get_tensor_from_cvol(filepath)
    if filepath.endswith('.h5'):
        return get_tensor_from_hdf5(filepath)


class IndexDataset(Dataset):
    def __init__(self, volume, sampleSize=16):
        self.vol_res = torch.tensor(volume.shape, dtype=torch.float)
        self.vol_res_touple = volume.shape
        self.n_voxels = torch.prod(self.vol_res).int().item()

        self.min_idx = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
        self.max_idx = torch.tensor([self.vol_res[0] - 1, self.vol_res[1] - 1, self.vol_res[2] - 1],
                                    dtype=torch.float) # M: can we just say vol_res -1?

        # M: transform into (nvoxel, 3) array for easy access
        self.volume_indices = self.generate_indices(self.min_idx, self.max_idx, self.vol_res.int()).view(-1, 3)

        self.sample_size = sampleSize
        self.max_dim = torch.max(self.max_idx)
        self.scales = self.max_idx / self.max_dim


    # M: generate a list with the indices of the volume-elements
    def generate_indices(self, start, end, res):
        positional_data = torch.zeros(res[0],res[1],res[2],3)

        positional_data[:,:,:,0] = torch.linspace(start[0],end[0],res[0],dtype=torch.float).view(res[0],1,1)
        positional_data[:,:,:,1] = torch.linspace(start[1],end[1],res[1],dtype=torch.float).view(1,res[1],1)
        positional_data[:,:,:,2] = torch.linspace(start[2],end[2],res[2],dtype=torch.float).view(1,1,res[2])

        return positional_data

    def move_data_to_device(self, device):
        self.min_idx = self.min_idx.to(device)
        self.max_idx = self.max_idx.to(device)
        self.vol_res = self.vol_res.to(device)
        self.scales = self.scales.to(device)
        self.volume_indices = self.volume_indices.to(device)

    def __len__(self):
        return self.n_voxels


    # M: get random positions in the volume for training
    def __getitem__(self, index):
        random_positions = self.volume_indices[torch.randint(0, self.n_voxels, (self.sample_size,))]
        normalized_positions = normalize_volume(random_positions,
                                                self.min_idx.unsqueeze(0), self.max_idx.unsqueeze(0),
                                                -1.0, 1.0)
        normalized_positions = self.scales.unsqueeze(0) * normalized_positions
        return random_positions, normalized_positions
