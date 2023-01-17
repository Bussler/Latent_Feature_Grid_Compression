import torch


class DropoutLayer(torch.nn.Module):

    def __init__(self, size=0, p: float = 0.5, threshold: float = 0.9):
        super().__init__()
        self.c = size
        self.p = p
        self.threshold = threshold

    def forward(self, x):
        pass

    def calculate_pruning_mask(self, device):
        pass

    @classmethod
    def create_instance(cls, size, sign_variance_momentum=0.02, threshold=0.9):
        return cls(size, sign_variance_momentum, threshold)
