import os.path
import torch
from data.IndexDataset import normalize_volume

#import common.utils # Could make failures in pyrenderer bib
import pyrenderer


def get_tensor_from_cvol(filepath):
    if not os.path.exists(filepath):
        raise ValueError(f'[ERROR] Volume file {filepath} does not exist.')
    vol = pyrenderer.Volume(filepath)
    feature = vol.get_feature(0)
    level = feature.get_level(0).to_tensor()
    volume = level.squeeze()

    # M: TODO Assert correct dimension!

    minV = torch.min(volume)
    maxV = torch.max(volume)
    volume = normalize_volume(volume, minV, maxV, -1.0, 1.0)

    print('Loaded CVOL Volume Successfully. Shape of: ', volume.shape)

    return volume
