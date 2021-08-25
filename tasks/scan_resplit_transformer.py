import dataset
from .scan_transformer import ScanTransformer


class ScanResplitTransformer(ScanTransformer):
    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.ScanLengthResplit("train", (0, self.helper.args.scan.length_cutoff))
        self.valid_sets.val = dataset.ScanLengthResplit("all", (self.helper.args.scan.length_cutoff+1, 9999))
        self.valid_sets.iid = dataset.ScanLengthResplit("test", (0, self.helper.args.scan.length_cutoff))
