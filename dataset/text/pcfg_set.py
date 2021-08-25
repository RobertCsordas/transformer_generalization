import os
from framework.utils import download
from .text_dataset import TextDataset, TextDatasetCache


class PCFGSet(TextDataset):
    URLS = {
        "simple": "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset",
        "productivity": "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/productivity",
        "substitutivity": "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/substitutivity/primitive",
        "systematicity": "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/systematicity",
    }

    def build_cache(self) -> TextDatasetCache:
        index_table = {}
        in_sentences = []
        out_sentences = []

        for split_type, url in self.URLS.items():
            index_table[split_type] = {}

            for set in ["test", "train"] + (["dev"] if split_type == "simple" else []):
                set_url = f"{url}/{set}"
                set_fn = os.path.join(self.cache_dir, split_type, os.path.split(set_url)[-1])
                os.makedirs(os.path.dirname(set_fn), exist_ok=True)

                for f in ["src", "tgt"]:
                    full_url = f"{set_url}.{f}"
                    print("Downloading", full_url)
                    download(full_url, f"{set_fn}.{f}", ignore_if_exists=True)

                this_set = []
                index_table[split_type][set] = this_set

                with open(set_fn + ".src") as f:
                    for line in f:
                        in_sentences.append(line.strip())
                        this_set.append(len(in_sentences) - 1)

                with open(set_fn + ".tgt") as f:
                    for line in f:
                        out_sentences.append(line.strip())

                assert len(in_sentences) == len(out_sentences)

        return TextDatasetCache().build(index_table, in_sentences, out_sentences)
