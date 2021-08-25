import torch
from dataclasses import dataclass
from .result import Result
from typing import List, Optional


@dataclass
class EncoderDecoderResult(Result):
    outputs: torch.Tensor
    out_lengths: torch.Tensor
    loss: torch.Tensor

    batch_dim = 1

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            return l[0]
        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.stack([r.outputs for r in l], l[0].batch_dim)
        lens = torch.stack([r.out_lengths for r in l], 0)
        return l[0].__class__(out, lens, loss)
