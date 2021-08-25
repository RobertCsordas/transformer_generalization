import dataset
from typing import Dict, Any
from .task import Task
from interfaces import Result
from .transformer_mixin import TransformerMixin
import framework


class COGSTransformer(TransformerMixin, Task):
    VALID_NUM_WORKERS = 0

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.COGS(["train"], shared_vocabulary=True)
        self.valid_sets.val = dataset.COGS(["valid"], shared_vocabulary=True)
        self.slow_valid_set = dataset.COGS(["gen"], shared_vocabulary=True)

    def create_loaders(self):
        super().create_loaders()
        self.slow_valid_loader = self.create_valid_loader(self.slow_valid_set)

    def do_generalization_test(self) -> Dict[str, Any]:
        d = {}
        test, loss = self.validate_on(self.slow_valid_set, self.slow_valid_loader)

        d["validation/gen/loss"] = loss
        d.update({f"validation/gen/{k}": v for k, v in test.plot().items()})
        d.update(self.update_best_accuracies("validation/gen", test.accuracy, loss))
        return d

    def plot(self, res: Result) -> Dict[str, Any]:
        d = super().plot(res)
        if (self.helper.state.iter % self.helper.args.cogs.generalization_test_interval == 0) or \
           (self.helper.state.iter == self.helper.args.test_interval):
            d.update(self.do_generalization_test())
        return d

    def train(self):
        super().train()
        if self.helper.state.iter % self.helper.args.cogs.generalization_test_interval != 0:
            # Redo the test, but only if it was not already done
            self.helper.summary.log(self.do_generalization_test())