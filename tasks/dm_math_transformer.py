import dataset
from .task import Task
from .transformer_mixin import TransformerMixin


class DMMathTransformer(TransformerMixin, Task):
    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.DeepmindMathDataset(self.helper.args.dm_math.tasks, sets=[f"train_{s}"
                                                     for s in self.helper.args.dm_math.train_splits])

        self.valid_sets.interpolate = dataset.DeepmindMathDataset(self.helper.args.dm_math.tasks, sets=["interpolate"])
        self.valid_sets.iid = dataset.DeepmindMathDataset(self.helper.args.dm_math.tasks, sets=[f"test_{s}" for s in
                                                          self.helper.args.dm_math.train_splits])

        extrapolate = dataset.DeepmindMathDataset(self.helper.args.dm_math.tasks, sets=["extrapolate"])
        if len(extrapolate) != 0:
            self.valid_sets["extrapolate"] = extrapolate
