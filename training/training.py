import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.Feature_Grid_Model import Feature_Grid_Model
from model.Feature_Embedding import FourierEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = None


def solve_model():
    pass


def training(args, verbose=True):
    # M: Get volume data, set up data

    # M: Setup Latent_Feature_Grid

    # M: Setup Embedder
    embedder = FourierEmbedding(n_freqs=2, input_dim=3)

    data = torch.ones((20,3))
    embedded_data = embedder.embed(data)

    # M: Setup model
    model = Feature_Grid_Model()
    model.to(device)
    model.train()

    pass


if __name__ == '__main__':
    args={}
    training(args)
