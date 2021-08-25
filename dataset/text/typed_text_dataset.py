from typing import Dict, List, Any
from .text_dataset import TextDataset, IndexTable, TextDatasetCache
import numpy as np


class TypedTextDatasetCache(TextDatasetCache):
    def build(self, index_table: IndexTable, in_sentences: List[str], out_sentences: List[str], types: List[int],
              type_names: List[str]):

        super().build(index_table, in_sentences, out_sentences)
        self.types = types
        self.type_names = type_names
        return self

    def state_dict(self) -> Dict[str, Any]:
        res = super().state_dict()
        res["types"] = self.types
        res["type_names"] = self.type_names
        return res

    def load_state_dict(self, state: Dict[str, Any]):
        super().load_state_dict(state)

        self.types = state["types"]
        self.type_names = state["type_names"]


class TypedTextDataset(TextDataset):
    _cache: TypedTextDatasetCache
    static_data: Dict[str, TypedTextDatasetCache] = {}

    def load_cache_file(self, file) -> TypedTextDatasetCache:
        return TypedTextDatasetCache.load(file)

    def build_cache(self) -> TypedTextDatasetCache:
        raise NotImplementedError()

    def __init__(self, sets: List[str] = ["train"], cache_dir: str = "./cache/", shared_vocabulary: bool = False):
        super().__init__(sets, ["default"], cache_dir, shared_vocabulary)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.my_indices[item]
        in_seq, out_seq = self.get_seqs(index)

        return {
            "in": np.asarray(in_seq, np.int16),
            "out": np.asarray(out_seq, np.int16),
            "in_len": len(in_seq),
            "out_len": len(out_seq),
            "type": self._cache.types[index]
        }
