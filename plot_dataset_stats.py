import dataset
from collections import OrderedDict
from multiprocessing import Process, Queue, cpu_count

datasets = OrderedDict()

datasets["0"] = None, None, None, None
datasets["Scan (length cutoff=26)"] = (dataset.ScanLengthResplit("train", (0, 26)),
                                     dataset.ScanLengthResplit("test", (0, 26)),
                                     dataset.ScanLengthResplit("all", (27, 9999)),
                                     None)


datasets["a"] = None, None, None, None
for i in range(1,4):
    datasets[f"CFQ MCD {i}"] = (dataset.CFQ(["train"], split_type=[f"mcd{i}"]),
                                None,
                                dataset.CFQ(["test"], split_type=[f"mcd{i}"]),
                                dataset.CFQ(["val"], split_type=[f"mcd{i}"]))

datasets["CFQ Output Length"]= (dataset.CFQ(["train"], split_type=[f"query_complexity"]),
                                None,
                                dataset.CFQ(["test"], split_type=[f"query_complexity"]),
                                dataset.CFQ(["val"], split_type=[f"query_complexity"]))


datasets["b"] = None, None, None, None
datasets["PCFG Productivity"]= (dataset.PCFGSet(["train"], split_type=["productivity"]),
                                None,
                                dataset.PCFGSet(["test"], split_type=["productivity"]),
                                None)

datasets["PCFG Systematicity"]= (dataset.PCFGSet(["train"], split_type=["systematicity"]),
                                None,
                                dataset.PCFGSet(["test"], split_type=["systematicity"]),
                                None)

datasets["c"] = None, None, None, None
datasets["COGS"] = (dataset.COGS(["train"]),
                    dataset.COGS(["valid"]),
                    dataset.COGS(["gen"]),
                    None)


datasets["d"] = None, None, None, None
datasets["Math: add\\_or\\_sub"] = (dataset.DeepmindMathDataset(["arithmetic__add_or_sub"],
                                                                sets=["train_easy", "train_medium", "train_hard"]),
                                    dataset.DeepmindMathDataset(["arithmetic__add_or_sub"], sets=["interpolate"]),
                                    dataset.DeepmindMathDataset(["arithmetic__add_or_sub"], sets=["extrapolate"]),
                                    None)

datasets["Math: place\\_value"] = (dataset.DeepmindMathDataset(["numbers__place_value"],
                                                                sets=["train_easy", "train_medium", "train_hard"]),
                                    dataset.DeepmindMathDataset(["numbers__place_value"], sets=["interpolate"]),
                                    dataset.DeepmindMathDataset(["numbers__place_value"], sets=["extrapolate"]),
                                    None)


def get_len(ds):
    nproc = cpu_count() * 2
    ranges = []
    prev = 0
    step = len(ds)//nproc
    for _ in range(nproc):
        next = prev + step
        ranges.append([prev, next])
        prev = next
    ranges[-1][-1] = len(ds)

    q = Queue()
    def cnt(r):
        mo = 0
        mi = 0
        for i in range(*r):
            mo = max(mo, ds[i]["out_len"])
            mi = max(mi, ds[i]["in_len"])
        q.put((mo, mi))

    procs = [Process(target=cnt, args=(r,)) for r in ranges]
    for p in procs:
        p.start()

    for p in procs:
        p.join()

    lens = [q.get() for _ in procs]
    return max([l[0] for l in lens]), max([l[1] for l in lens])

print("Dataset & \\# train & \\# IID valid. & \\# gen. test & \\# gen. valid. & Voc. size & Train len. & Test len.\\\\")
for name, dsdesc in datasets.items():
    train, val_iid, test_gen, val_gen = dsdesc
    if train is None:
        print("\\midrule")
        continue

    print(f"{name} ", end="")
    for ds in dsdesc:
        print(" & " + (f"${len(ds)}$" if ds is not None else '-'), end="")

    max_train_out, max_train_in = get_len(train)
    max_test_out, max_test_in = get_len(test_gen)
    print(f" & ${len(train.in_vocabulary+train.out_vocabulary)}$ & ${max_train_in}$/${max_train_out}$ & ${max_test_in}$/${max_test_out}$ \\\\")
