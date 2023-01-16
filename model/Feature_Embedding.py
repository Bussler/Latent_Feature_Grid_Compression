import torch
import numpy as np


# M: class to embed input into higher-level frequency space
class Embedder:
    def __init__(self):
        self.embed_functions = []
        self.out_dim = 0

    def create_embedding_function(self):
        pass

    def embed(self, inputs):
        embedded = torch.cat([fn(inputs) for fn in self.embed_functions], -1)  # M: TODO refactor? All Freqbands in one matrix and cat for sin/ cos parts
        return embedded


# M: Fourier embedding like described in Weiss fv-SRN paper
class FourierEmbedding(Embedder):
    def __init__(self, n_freqs, input_dim):
        super(FourierEmbedding, self).__init__()

        self.periodic_functions = [torch.sin, torch.cos]
        self.create_embedding_function(n_freqs, input_dim)

    def create_embedding_function(self, n_freqs, input_dim):
        freq_bands = 2. ** torch.linspace(0., n_freqs-1, steps=n_freqs)
        freq_bands = freq_bands * 2. * np.pi

        for freq in freq_bands:
            for p_fn in self.periodic_functions:
                self.embed_functions.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                self.out_dim += input_dim
