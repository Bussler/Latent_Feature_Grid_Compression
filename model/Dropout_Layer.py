import torch


class DropoutLayer(torch.nn.Module):

    i = 0
    theshold_list = None

    def __init__(self, size=0, p: float = 0.5, threshold: float = 0.9):
        super().__init__()
        self.c = size
        self.p = p
        self.threshold = threshold

        if DropoutLayer.theshold_list is not None:  #M used for testing, needs refactoring
            if DropoutLayer.i != 0:
                self.threshold = DropoutLayer.theshold_list[DropoutLayer.i-1]
            DropoutLayer.i = DropoutLayer.i + 1

    def forward(self, x):
        pass

    def calculate_pruning_mask(self, device):
        pass

    def multiply_values_with_dropout(self, input, device):
        pass

    @classmethod
    def set_threshold_list(cls, list: []):
        cls.i = 0
        cls.theshold_list = list

    @classmethod
    def create_instance(cls, size, sign_variance_momentum=0.02, threshold=0.9):
        return cls(size, sign_variance_momentum, threshold)
