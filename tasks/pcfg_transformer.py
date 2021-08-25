import dataset
from typing import Dict, Any
from .task import Task
from .transformer_mixin import TransformerMixin


class PCFGTransformer(TransformerMixin, Task):
    VALID_NUM_WORKERS = 0
    MAX_LENGHT_PER_BATCH = 200

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.PCFGSet(["train"], split_type=[self.helper.args.pcfg.split], shared_vocabulary=True)
        self.valid_sets.val = dataset.PCFGSet(["test"], split_type=[self.helper.args.pcfg.split],
                                              shared_vocabulary=True)
