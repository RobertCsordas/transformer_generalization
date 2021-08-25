import torch
import torch.nn


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
    input = torch.cat((input, torch.zeros_like(input[0:1])), dim=0)
    input.scatter_(0, lengths.unsqueeze(0).long(), value=eos_id)
    return input

