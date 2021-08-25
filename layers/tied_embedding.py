import torch
import torch.nn
import torch.nn.functional as F


class TiedEmbedding(torch.nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()

        # Hack: won't save it as a parameter
        self.w = [weights]
        self.bias = torch.nn.Parameter(torch.zeros(self.w[0].shape[0]))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.linear(t, self.w[0], self.bias)
