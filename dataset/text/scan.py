import os
from framework.utils import download
from .text_dataset import TextDataset, TextDatasetCache


class Scan(TextDataset):
    URLS = {
        "simple": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt"
        },
        "length": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"
        },
        "jump": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt"
        },
        "turn_left": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_turn_left.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_turn_left.txt"
        },
    }

    def build_cache(self) -> TextDatasetCache:
        index_table = {}
        in_sentences = []
        out_sentences = []

        for split_type, split in self.URLS.items():
            index_table[split_type] = {}

            for set, url in split.items():
                fn = os.path.join(self.cache_dir, os.path.split(url)[-1])

                print("Downloading", url)
                download(url, fn, ignore_if_exists=True)

                this_set = []
                index_table[split_type][set] = this_set

                with open(fn) as f:
                    for line in f:
                        line = line.split("OUT:")
                        line[0] = line[0].replace("IN:", "")
                        line = [l.strip() for l in line]

                        in_sentences.append(line[0])
                        out_sentences.append(line[1])

                        this_set.append(len(in_sentences) - 1)

        return TextDatasetCache().build(index_table, in_sentences, out_sentences)
