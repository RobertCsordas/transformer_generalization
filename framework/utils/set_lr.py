import torch


def set_lr(optim: torch.optim.Optimizer, lr: float):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
