import dataset
from .task import Task
from .transformer_mixin import TransformerMixin
from typing import Tuple, Any
import framework
import dataset.sequence
import numpy as np
import torch


class CFQTransformer(TransformerMixin, Task):
    VALID_NUM_WORKERS = 0
    MAX_LENGHT_PER_BATCH = 101

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.CFQ(["train"], split_type=[self.helper.args.cfq.split])
        self.valid_sets.val = dataset.CFQ(["val"], split_type=[self.helper.args.cfq.split])
        self.valid_sets.test = dataset.CFQ(["test"], split_type=[self.helper.args.cfq.split])

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)
        self.init_valid_details()

    def init_valid_details(self):
        self.helper.state.full_loss_log = {}

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        res, loss = self.validate_on(self.valid_sets[name], self.valid_loaders[name])
        if self.helper.args.log_sample_level_loss and isinstance(res, dataset.sequence.TextSequenceTestState):
            losses, oks = res.get_sample_info()
            if name not in self.helper.state.full_loss_log:
                self.helper.state.full_loss_log[name] = [], []

            self.helper.state.full_loss_log[name][0].append(losses)
            self.helper.state.full_loss_log[name][1].append(oks)

        return res, loss

    def save_valid_details(self):
        for name, (losses, oks) in self.helper.state.full_loss_log.items():
            losses = np.asfarray(losses)
            oks = np.asarray(oks, dtype=np.bool)
            torch.save({"losses": losses, "oks": oks}, self.helper.get_storage_path(f"loss_details/{name}.pth"))

    def train(self):
        super().train()
        self.save_valid_details()
