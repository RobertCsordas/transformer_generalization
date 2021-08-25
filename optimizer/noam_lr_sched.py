class NoamLRSched:
    def __init__(self, lr: float, state_size: int, warmup_steps: int):
        self.lr = lr / (state_size ** 0.5)
        self.warmup_steps = warmup_steps

    def get(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.lr / float(step + 1) ** 0.5
        else:
            return self.lr / (self.warmup_steps**1.5) * float(step + 1)
