import torch
import torch.utils.data
import os
import numpy as np
from framework.utils import download
from framework.data_structures import WordVocabulary
from typing import Dict, Any, Tuple
from .sequence import TextSequenceTestState


class ScanLengthResplit(torch.utils.data.Dataset):
    in_sentences = []
    out_sentences = []
    index_table = {}

    URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/tasks.txt"

    def _load_dataset(self, cache_dir: str):
        if ScanLengthResplit.in_sentences:
            return

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "scan.pth")

        if not os.path.isfile(cache_file):
            fn = os.path.join(cache_dir, os.path.split(self.URL)[-1])

            print("Downloading", self.URL)
            download(self.URL, fn, ignore_if_exists=True)

            with open(fn) as f:
                for line in f:
                    line = line.split("OUT:")
                    line[0] = line[0].replace("IN:", "")
                    line = [l.strip() for l in line]

                    ScanLengthResplit.in_sentences.append(line[0])
                    ScanLengthResplit.out_sentences.append(line[1])

            print("Constructing vocabularies")
            ScanLengthResplit.in_vocabulary = WordVocabulary(self.in_sentences)
            ScanLengthResplit.out_vocabulary = WordVocabulary(self.out_sentences)

            ScanLengthResplit.in_sentences = [ScanLengthResplit.in_vocabulary(s)
                                              for s in ScanLengthResplit.in_sentences]
            ScanLengthResplit.out_sentences = [ScanLengthResplit.out_vocabulary(s)
                                               for s in ScanLengthResplit.out_sentences]

            ScanLengthResplit.max_in_len = max(len(l) for l in ScanLengthResplit.in_sentences)
            ScanLengthResplit.max_out_len = max(len(l) for l in ScanLengthResplit.out_sentences)

            print("Done.")
            torch.save({
                "in_sentences": ScanLengthResplit.in_sentences,
                "out_sentences": ScanLengthResplit.out_sentences,
                "in_voc": ScanLengthResplit.in_vocabulary.state_dict(),
                "out_voc": ScanLengthResplit.out_vocabulary.state_dict(),
                "max_in_len": ScanLengthResplit.max_in_len,
                "max_out_len": ScanLengthResplit.max_out_len
            }, cache_file)
        else:
            data = torch.load(cache_file)
            ScanLengthResplit.in_vocabulary = WordVocabulary(None)
            ScanLengthResplit.out_vocabulary = WordVocabulary(None)
            ScanLengthResplit.in_vocabulary.load_state_dict(data["in_voc"])
            ScanLengthResplit.out_vocabulary.load_state_dict(data["out_voc"])
            ScanLengthResplit.in_sentences = data["in_sentences"]
            ScanLengthResplit.out_sentences = data["out_sentences"]
            ScanLengthResplit.max_in_len = data["max_in_len"]
            ScanLengthResplit.max_out_len = data["max_out_len"]


    def __init__(self, dset: str, len_range: Tuple[int, int], train_proprtion: float = 0.9,
                 cache_dir: str = "./cache/scan_resplit"):
        super().__init__()
        self.cache_dir = cache_dir
        self._load_dataset(cache_dir)
        self.len_range = len_range

        assert dset in ["train", "test", "all"]

        self.my_indices = [i for i, o in enumerate(self.out_sentences) if len_range[0] <= len(o) <= len_range[1]]

        if dset != "all":
            seed = np.random.RandomState(1234)
            test_indices = set(seed.choice(len(self.my_indices), int(len(self.my_indices) * (1 - train_proprtion)),
                                           replace=False).tolist())

            self.my_indices = [i for ii, i in enumerate(self.my_indices) if (ii in test_indices) ^ (dset == "train")]

        self.this_max_out_len = max(len(self.out_sentences[i]) for i in self.my_indices)
        self.this_min_out_len = min(len(self.out_sentences[i]) for i in self.my_indices)

    def __len__(self) -> int:
        return len(self.my_indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.my_indices[item]
        in_seq = ScanLengthResplit.in_sentences[index]
        out_seq = ScanLengthResplit.out_sentences[index]

        return {
            "in": np.asarray(in_seq, np.int16),
            "out": np.asarray(out_seq, np.int16),
            "in_len": len(in_seq),
            "out_len": len(out_seq)
        }

    def get_output_size(self):
        return len(self.out_vocabulary)

    def get_input_size(self):
        return len(self.in_vocabulary)

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                     lambda x: " ".join(self.out_vocabulary(x)))

    def __str__(self):
        return f"ScanLengthResplit(range=[{self.this_min_out_len}, {self.this_max_out_len}], len={len(self)})"

    __repr__ = __str__
