import torch
from typing import Union, Any, Dict


class Average:
    SAVE = ["sum", "cnt"]

    def __init__(self):
        self.reset()

    def add(self, data: Union[int, float, torch.Tensor]):
        if torch.is_tensor(data):
            data = data.detach()

        self.sum += data
        self.cnt += 1

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def get(self, reset=True) -> Union[float, torch.Tensor]:
        res = self.sum / self.cnt
        if reset:
            self.reset()

        return res

    def state_dict(self) -> Dict[str, Any]:
        return {k: self.__dict__[k] for k in self.SAVE}

    def load_state_dict(self, state: Dict[str, Any]):
        self.__dict__.update(state or {})


class MovingAverage(Average):
    SAVE = ["sum", "cnt", "history"]

    def __init__(self, window_size: int):
        self.window_size = window_size
        super().__init__()

    def reset(self):
        self.history = []
        super().reset()

    def add(self, data: Union[int, float, torch.Tensor]):
        super().add(data)
        if self.cnt > self.window_size:
            self.sum -= self.history.pop(0)
            self.cnt -= 1

        assert self.cnt <= self.window_size
