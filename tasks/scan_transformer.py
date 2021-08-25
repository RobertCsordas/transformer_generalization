import dataset
from .task import Task
from .transformer_mixin import TransformerMixin


class ScanTransformer(TransformerMixin, Task):
    VALID_NUM_WORKERS = 0

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Scan(["train"], split_type=self.helper.args.scan.train_split)
        self.valid_sets.val = dataset.Scan(["test"], split_type=self.helper.args.scan.train_split)
        self.valid_sets.iid = dataset.Scan(["test"])
