import os
import tarfile
from typing import List, Tuple, Callable, Dict, Any, Optional
import framework
import numpy as np
import itertools
from ..sequence import TypedTextSequenceTestState
import pickle
import gc
import shutil
import bisect
import torch.utils.data


# Not using text dataset, because it is too big. Uses mmap and some tricks to save memory.
class DeepmindMathDataset(torch.utils.data.Dataset):
    VERSION = 8

    vocabulary: framework.data_structures.CharVocabulary = None
    raw_data = {}
    index = {}
    DIFFICULTIES = ["easy", "medium", "hard"]

    def lock(self) -> framework.utils.LockFile:
        return framework.utils.LockFile(os.path.join(self.cache_dir, "dm_math_lock"))

    def download(self):
        with self.lock():
            os.makedirs(self.cache_dir, exist_ok=True)
            if not os.path.isdir(os.path.join(self.cache_dir, "mathematics_dataset-v1.0")):
                if not os.path.isfile(os.path.join(self.cache_dir, "mathematics_dataset-v1.0.tar.gz")):
                    assert False, "Please download https://console.cloud.google.com/storage/browser/_details/"\
                                  "mathematics-dataset/mathematics_dataset-v1.0.tar.gz and place it in the"\
                                  f" {os.path.abspath(self.cache_dir)} folder."

                with tarfile.open(os.path.join(self.cache_dir, "mathematics_dataset-v1.0.tar.gz"), "r") as tf:
                    tf.extractall(path=self.cache_dir)

    def load_file(self, path: str) -> Tuple[List[str], List[str]]:
        print(f"Loading {path}")
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines()]

        q = lines[::2]
        a = lines[1::2]
        assert len(q) == len(a)
        return q, a

    def verify_cache_version(self):
        with self.lock():
            if os.path.isfile(self.version_cache):
                verfile = pickle.load(open(self.version_cache, 'rb'))
                if verfile["version"] == self.VERSION:
                    return

            # Create new cache
            print("Cache version changed. Invalidating the cache...")
            shutil.rmtree(self.task_cache, ignore_errors=True)

            if os.path.exists(self.vocab_cache):
                os.remove(self.vocab_cache)
            pickle.dump({"version": self.VERSION}, open(self.version_cache, "wb"))

    def get_cached(self, fname: str, construct: Callable[[], Any]) -> Any:
        # Only one process can check for existence and load the file at once.
        with self.lock():
            if not os.path.isfile(fname):
                data = construct()
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                gc.disable()
                pickle.dump(data, open(fname, "wb"), protocol=-1)
                gc.enable()
                return data

        # Load
        gc.disable()
        data = pickle.load(open(fname, "rb"))
        gc.enable()

        return data

    def create_vocab(self) -> set:
        print("Constructing vocabulary...")

        flist = []
        extracted_dir = os.path.join(self.cache_dir, "mathematics_dataset-v1.0")
        for s in os.listdir(extracted_dir):
            if "readme" in s:
                continue

            set_dir = os.path.join(self.cache_dir, "mathematics_dataset-v1.0", s)

            for task in os.listdir(set_dir):
                flist.append(os.path.join(set_dir, task))


        def process(fname: str):
            vocabulary = set()
            questions, answers = self.load_file(fname)
            for q in questions:
                vocabulary.update(set(q))

            for a in answers:
                vocabulary.update(set(a))

            return vocabulary

        vlist = framework.utils.parallel_map(flist, process)
        return set().union(*vlist)

    def translate_file(self, fname: str, file, known: set):
        print(f"Translating {fname}")
        questions, answers = self.load_file(fname)

        index = []
        cache = []
        offset = file.tell()

        skipped = 0

        def sync():
            np.asarray(list(itertools.chain.from_iterable(cache)), dtype=np.int8).astype('int8').tofile(file)
            assert offset == file.tell()
            cache.clear()

        for q, a in zip(questions, answers):
            h = hash(q)
            if h in known:
                skipped += 1
                continue
            else:
                known.add(h)

            cache.append(self.vocabulary(q))
            cache.append(self.vocabulary(a))
            len_total = len(q)+len(a)
            index.append([offset, len_total, len(q)])
            offset += len_total
            if len(cache)>10000:
                sync()

        if skipped>0:
            print(f"WARNING: removed {skipped} entries from {fname} because of repeats.")

        if len(cache)>0:
            sync()
        return index

    def get_task_name_list(self) -> List[str]:
        extracted_dir = os.path.join(self.cache_dir, "mathematics_dataset-v1.0")
        res = set()
        for s in os.listdir(extracted_dir):
            if "readme" in s:
                continue

            is_train = s.startswith("train-")
            tname = s[6:] if is_train else s

            for f in os.listdir(os.path.join(extracted_dir, s)):
                assert f.endswith(".txt")
                f = f[:-4]

                if tname == "extrapolate":
                    for e in ["_big", "_more", "_longer"]:
                        i = f.find(e)
                        if i > 0:
                            f = f[:i]
                            break
                    
                if is_train:
                    res.add(f"{f}_train_{tname}")
                    res.add(f"{f}_test_{tname}")
                else:
                    res.add(f"{f}_{tname}")
        return list(sorted(res))

    def split_test(self, data: List) -> Tuple[List, List]:
        def copy_filtered(data: List, filter) -> List:
            return [data[i] for i in range(len(data)) if filter(i)]

        seed = np.random.RandomState(1234)
        test_indices = set(seed.choice(len(data), 10000, replace=False).tolist())

        return copy_filtered(data, lambda i: i not in test_indices),\
               copy_filtered(data, lambda i: i in test_indices)

    def write_index_list(self, fname:str, ilist: List[List]):
        f = open(os.path.join(self.task_cache, fname), "wb")
        for l in ilist:
            np.asarray(l, dtype=np.uint32).astype('uint32').tofile(f)

    def collect_missing_tasks(self, tasks: List[str]) -> List[str]:
        return [t for t in tasks if not os.path.isfile(os.path.join(self.task_cache, t + ".raw"))]

    def find_extrapolation_set(self, task: str) -> Optional[str]:
        extrapolate_dir = os.path.join(self.cache_dir, "mathematics_dataset-v1.0", f"extrapolate")
        found = None
        for f in os.listdir(extrapolate_dir):
            if f.startswith(task):
                assert found is None, f"Multiple extrapolation sets found for task {task}"
                found = os.path.join(extrapolate_dir, f)
        return found

    def create_task_cache(self, task: str):
        raw_file = open(os.path.join(self.task_cache, task + ".raw"), "wb")

        data = [os.path.join(self.cache_dir, "mathematics_dataset-v1.0", f"train-{k}", task + ".txt")
                for k in self.DIFFICULTIES]
        data.append(os.path.join(self.cache_dir, "mathematics_dataset-v1.0", f"interpolate", task + ".txt"))
        extrapolation = self.find_extrapolation_set(task)
        if extrapolation is not None:
            print(f"Found extrapolation set {extrapolation}")
            data.append(extrapolation)

        known = set()
        data = [self.translate_file(d, raw_file, known) for d in data]
        data = [self.split_test(d) if i<len(self.DIFFICULTIES) else d for i, d in enumerate(data)]

        for i, (n, d) in enumerate(zip(self.DIFFICULTIES, data)):
            self.write_index_list(f"{task}_train_{n}.idx", d[0])
            self.write_index_list(f"{task}_test_{n}.idx", d[1])

        self.write_index_list(f"{task}_interpolate.idx", data[len(self.DIFFICULTIES)])
        if extrapolation is not None:
            self.write_index_list(f"{task}_extrapolate.idx", data[len(self.DIFFICULTIES)+1])

    def list_task_indices(self, task: str):
        return [file[len(task)+1:-4] for file in os.listdir(self.task_cache)
                if file.endswith(".idx") and file.startswith(task)]

    def load_task(self, task: str):
        if task in DeepmindMathDataset.raw_data:
            return

        DeepmindMathDataset.raw_data[task] = np.memmap(os.path.join(self.task_cache, task+ ".raw"), dtype='uint8',
                                                       mode='r')
        DeepmindMathDataset.index[task] = {
            n: np.memmap(os.path.join(self.task_cache, task+f"_{n}.idx"), dtype='uint32', mode="r") \
               for n in self.list_task_indices(task)
        }

    def load_vocab(self):
        if DeepmindMathDataset.vocabulary is not None:
            return

        vocabulary = self.get_cached(self.vocab_cache, self.create_vocab)
        DeepmindMathDataset.vocabulary = framework.data_structures.CharVocabulary(vocabulary)
        DeepmindMathDataset.in_vocabulary = DeepmindMathDataset.vocabulary
        DeepmindMathDataset.out_vocabulary = DeepmindMathDataset.vocabulary

        print(f"Vocabulary: {len(vocabulary)}")
        print(vocabulary)

    def __len__(self) -> int:
        return self.count

    def __init__(self, tasks: List[str], sets: List[str] = ["train_easy", "train_medium", "train_hard"],
                 cache_dir: str="./cache/dm_math/"):

        super().__init__()
        self.cache_dir = cache_dir
        self.vocab_cache = os.path.join(self.cache_dir, "vocabulary.dat")
        self.version_cache = os.path.join(self.cache_dir, "version.dat")
        self.task_cache = os.path.join(self.cache_dir, "cached_tasks")

        os.makedirs(self.cache_dir, exist_ok=True)

        self.download()
        self.verify_cache_version()

        os.makedirs(self.task_cache, exist_ok=True)

        self.data_tables = []
        self.index_tables = []
        self.offset_table = []
        self.table_type = []
        self.count = 0

        self.task_list = self.get_task_name_list()

        print("Loading vocabulary")
        self.load_vocab()

        with self.lock():
            missing_tasks = self.collect_missing_tasks(tasks)
            framework.utils.parallel_map(missing_tasks, self.create_task_cache, max_parallel=16)

        for t in tasks:
            self.load_task(t)

            for set in sets:
                print(f"Loading task {t}, set {set}")
                if set=="extrapolate" and set not in DeepmindMathDataset.index[t]:
                    print(f"No extrapolation set for {t}")
                    continue

                self.table_type.append(self.task_list.index(f"{t}_{set}"))
                self.index_tables.append(DeepmindMathDataset.index[t][set])
                self.data_tables.append(DeepmindMathDataset.raw_data[t])
                self.offset_table.append(self.count)
                self.count += self.index_tables[-1].shape[0] // 3

        print(f"Loaded {len(self)} samples")

    def get_index(self, item: int) -> Tuple[np.ndarray, np.ndarray, int]:
        table_index = bisect.bisect(self.offset_table, item) - 1
        relative_index = item - self.offset_table[table_index]

        offset, t_len, q_len = self.index_tables[table_index][relative_index*3:(relative_index+1)*3]
        return self.data_tables[table_index][offset:offset+q_len], \
               self.data_tables[table_index][offset+q_len:offset+t_len], self.table_type[table_index]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        q, a, tid = self.get_index(item)
        return {
            "in": q,
            "out": a,
            "in_len": q.shape[0],
            "out_len": a.shape[0],
            "type": tid
        }

    def start_test(self) -> TypedTextSequenceTestState:
        return TypedTextSequenceTestState(self.in_vocabulary.ind_to_str, self.out_vocabulary.ind_to_str, self.task_list)
