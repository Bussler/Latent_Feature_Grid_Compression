import torch


class DropoutLayer(torch.nn.Module):

    def __init__(self, p: float = 0.5, threshold: float = 0.9):
        super().__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, x):
        pass

    def calculate_pruning_mask(self, device):
        pass

    @classmethod
    def create_instance(cls):
        pass
