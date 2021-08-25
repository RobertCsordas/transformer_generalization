import torch
import torch.utils.data
import os
import numpy as np
import framework
from framework.data_structures import WordVocabulary
from typing import List, Dict, Any, Tuple
from ..sequence import TextSequenceTestState

IndexTable = Dict[str, Dict[str, List[int]]]
VERSION = 5


class TextDatasetCache:
    version: int
    in_sentences: List[List[int]]
    out_sentences: List[List[int]]

    index_table: IndexTable
    in_vocabulary: WordVocabulary
    out_vocabulary: WordVocabulary

    len_histogram: Dict[float, int]

    max_in_len: int
    max_out_len: int

    def build(self, index_table: IndexTable, in_sentences: List[str], out_sentences: List[str],
              split_punctuation: bool = True):
        self.version = VERSION
        self.index_table = index_table

        print("Constructing vocabularies")
        self.in_vocabulary = WordVocabulary(in_sentences, split_punctuation=split_punctuation)
        self.out_vocabulary = WordVocabulary(out_sentences, split_punctuation=split_punctuation)

        self.in_sentences = [self.in_vocabulary(s) for s in in_sentences]
        self.out_sentences = [self.out_vocabulary(s) for s in out_sentences]

        print("Calculating length statistics")
        counts, bins = np.histogram([len(i)+len(o) for i, o in zip(self.in_sentences, self.out_sentences)])
        self.sum_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        counts, bins = np.histogram([len(i) for i in self.in_sentences])
        self.in_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        counts, bins = np.histogram([len(o) for o in self.out_sentences])
        self.out_len_histogram = {k: v for k, v in zip(bins.tolist(), counts.tolist())}

        self.max_in_len = max(len(l) for l in self.in_sentences)
        self.max_out_len = max(len(l) for l in self.out_sentences)
        print("Done.")

        return self

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "index": self.index_table,
            "in_sentences": self.in_sentences,
            "out_sentences": self.out_sentences,
            "in_voc": self.in_vocabulary.state_dict(),
            "out_voc": self.out_vocabulary.state_dict(),
            "max_in_len": self.max_in_len,
            "max_out_len": self.max_out_len,
            "in_len_histogram": self.in_len_histogram,
            "sum_len_histogram": self.sum_len_histogram,
            "out_len_histogram": self.out_len_histogram
        }

    def load_state_dict(self, data: Dict[str, Any]):
        self.version = data.get("version", -1)
        if self.version != VERSION:
            return
        self.index_table = data["index"]
        self.in_vocabulary = WordVocabulary(None)
        self.out_vocabulary = WordVocabulary(None)
        self.in_vocabulary.load_state_dict(data["in_voc"])
        self.out_vocabulary.load_state_dict(data["out_voc"])
        self.in_sentences = data["in_sentences"]
        self.out_sentences = data["out_sentences"]
        self.max_in_len = data["max_in_len"]
        self.max_out_len = data["max_out_len"]
        self.in_len_histogram = data["in_len_histogram"]
        self.out_len_histogram = data["out_len_histogram"]
        self.sum_len_histogram = data["sum_len_histogram"]

    def save(self, fn: str):
        torch.save(self.state_dict(), fn)

    @classmethod
    def load(cls, fn: str):
        res = cls()
        try:
            data = torch.load(fn)
        except:
            print(f"Failed to load cache file. {fn}")
            res.version = -1
            return res

        res.load_state_dict(data)

        return res


class TextDataset(torch.utils.data.Dataset):
    static_data: Dict[str, TextDatasetCache] = {}

    def build_cache(self) -> TextDatasetCache:
        raise NotImplementedError()

    def load_cache_file(self, file) -> TextDatasetCache:
        return TextDatasetCache.load(file)

    def _load_dataset(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "cache.pth")
        
        if os.path.isfile(cache_file):
            res = self.load_cache_file(cache_file)
            if res.version == VERSION:
                return res
            else:
                print(f"{self.__class__.__name__}: Invalid cache version: {res.version}, current: {VERSION}")

        with framework.utils.LockFile(os.path.join(self.cache_dir, "lock")):
            res = self.build_cache()
            res.save(cache_file)
        return res

    def hist_to_text(self, histogram: Dict[float, int]) -> str:
        keys = list(sorted(histogram.keys()))
        values = [histogram[k] for k in keys]
        percent = (np.cumsum(values) * (100.0 / sum(histogram.values()))).tolist()
        return ", ".join(f"{k:.1f}: {v} (>= {p:.1f}%)" for k, v, p in zip(keys, values, percent))
     
    def __init__(self, sets: List[str] = ["train"], split_type: List[str] = ["simple"], cache_dir: str = "./cache/",
                 shared_vocabulary: bool = False):
        super().__init__()

        self.cache_dir = os.path.join(cache_dir, self.__class__.__name__)
        os.makedirs(self.cache_dir, exist_ok=True)

        assert isinstance(sets, List)
        assert isinstance(split_type, List)

        self._cache = TextDataset.static_data.get(self.__class__.__name__)
        just_loaded = self._cache is None
        if just_loaded:
            self._cache = self._load_dataset()
            TextDataset.static_data[self.__class__.__name__] = self._cache

        if shared_vocabulary:
            self.in_vocabulary = self._cache.in_vocabulary + self._cache.out_vocabulary
            self.out_vocabulary = self.in_vocabulary
            self.in_remap = self.in_vocabulary.mapfrom(self._cache.in_vocabulary)
            self.out_remap = self.out_vocabulary.mapfrom(self._cache.out_vocabulary)
        else:
            self.in_vocabulary = self._cache.in_vocabulary
            self.out_vocabulary = self._cache.out_vocabulary

        if just_loaded:
            for k, t in self._cache.index_table.items():
                print(f"{self.__class__.__name__}: split {k} data:",
                      ", ".join([f"{k}: {len(v)}" for k, v in t.items()]))
            print(f"{self.__class__.__name__}: vocabulary sizes: in: {len(self._cache.in_vocabulary)}, "
                  f"out: {len(self._cache.out_vocabulary)}")
            print(f"{self.__class__.__name__}: max input length: {self._cache.max_in_len}, "
                  f"max output length: {self._cache.max_out_len}")
            print(f"{self.__class__.__name__} sum length histogram: {self.hist_to_text(self._cache.sum_len_histogram)}")
            print(f"{self.__class__.__name__} in length histogram: {self.hist_to_text(self._cache.in_len_histogram)}")
            print(f"{self.__class__.__name__} out length histogram: {self.hist_to_text(self._cache.out_len_histogram)}")

        self.my_indices = []
        for t in split_type:
            for s in sets:
                self.my_indices += self._cache.index_table[t][s]

        self.shared_vocabulary = shared_vocabulary

    def get_seqs(self, abs_index: int) -> Tuple[List[int], List[int]]:
        in_seq = self._cache.in_sentences[abs_index]
        out_seq = self._cache.out_sentences[abs_index]

        if self.shared_vocabulary:
            in_seq = [self.in_remap[i] for i in in_seq]
            out_seq = [self.out_remap[i] for i in out_seq]

        return in_seq, out_seq

    def __len__(self) -> int:
        return len(self.my_indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.my_indices[item]
        in_seq, out_seq = self.get_seqs(index)

        return {
            "in": np.asarray(in_seq, np.int16),
            "out": np.asarray(out_seq, np.int16),
            "in_len": len(in_seq),
            "out_len": len(out_seq)
        }

    def get_output_size(self):
        return len(self._cache.out_vocabulary)

    def get_input_size(self):
        return len(self._cache.in_vocabulary)

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)), 
                                     lambda x: " ".join(self.out_vocabulary(x)))

    @property
    def max_in_len(self) -> int:
        return self._cache.max_in_len

    @property
    def max_out_len(self) -> int:
        return self._cache.max_out_len
