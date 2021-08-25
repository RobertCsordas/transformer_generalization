import torch
import torch.nn.functional as F
from typing import Optional
import math


def cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean", smoothing: float = 0,
                  ignore_index: Optional[int] = None) -> torch.Tensor:

    # Flatten inputs to 2D
    t2 = target.flatten().long()
    i2 = input.flatten(end_dim=-2)

    # If no smoothing, use built-in cross_entropy loss
    if smoothing == 0:
        loss = F.cross_entropy(i2, t2, reduction=reduction, ignore_index=-100 if ignore_index is None else ignore_index)
        if reduction == "none":
            return loss.view_as(target)
        else:
            return loss
    
    # Calculate the softmax cross entropy loss
    i2 = F.log_softmax(i2, -1)
    right_class = i2.gather(-1, t2.unsqueeze(-1)).squeeze()
    others = i2.sum(-1) - right_class

    # KL divergence
    loss = (smoothing - 1.0) * right_class - others * smoothing
    optimal_loss = -((1.0 - smoothing) * math.log(1 - smoothing) + (i2.shape[1] - 1) * smoothing * math.log(smoothing))

    loss = loss - optimal_loss

    # Handle masking if igonore_index is specified
    if ignore_index is not None:
        tmask = t2 != ignore_index
        loss = torch.where(tmask, loss, torch.zeros([1], dtype=loss.dtype, device=loss.device))
        n_total = tmask.float().sum()
    else:
        n_total = t2.nelement()

    # Reduction
    if reduction == "none":
        return loss.view_as(target)
    elif reduction == "mean":
        return loss.sum() / n_total
    elif reduction == "sum":
        return loss.sum()
    else:
        assert False, f"Invalid reduction {reduction}"
