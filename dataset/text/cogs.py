import os
from framework.utils import download
import csv
from .typed_text_dataset import TypedTextDataset, TypedTextDatasetCache
from ..sequence import TypedTextSequenceTestState


class COGS(TypedTextDataset):
    URL_BASE = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/"
    SPLT_TYPES = ["train", "test", "valid", "gen"]
    NAME_MAP = {"valid": "dev"}

    def build_cache(self) -> TypedTextDatasetCache:

        types = []
        type_list = []
        type_map = {}

        index_table = {}
        in_sentences = []
        out_sentences = []

        for st in self.SPLT_TYPES:
            fname = self.NAME_MAP.get(st, st) + ".tsv"
            split_fn = os.path.join(self.cache_dir, fname)
            os.makedirs(os.path.dirname(split_fn), exist_ok=True)

            full_url = self.URL_BASE + fname
            print("Downloading", full_url)
            download(full_url, split_fn, ignore_if_exists=True)

            index_table[st] = []

            with open(split_fn, "r") as f:
                d = csv.reader(f, delimiter="\t")
                for line in d:
                    i, o, t = line

                    index_table[st].append(len(in_sentences))
                    in_sentences.append(i)
                    out_sentences.append(o)

                    tind = type_map.get(t)
                    if tind is None:
                        type_map[t] = tind = len(type_list)
                        type_list.append(t)

                    types.append(tind)

                assert len(in_sentences) == len(out_sentences)

        return TypedTextDatasetCache().build({"default": index_table}, in_sentences, out_sentences, types, type_list)

    def start_test(self) -> TypedTextSequenceTestState:
        return TypedTextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                          lambda x: " ".join(self.out_vocabulary(x)),
                                          self._cache.type_names)
