from torch import multiprocessing
from typing import Iterable, Callable, Any, List
import time


def parallel_map(tasks: Iterable, callback = Callable[[Any], None], max_parallel: int = 32) -> List:
    limit = min(multiprocessing.cpu_count(), max_parallel)
    processes: List[multiprocessing.Process] = []
    queues: List[multiprocessing.Queue] = []
    indices: List[int] = []
    tlist = [t for t in tasks]
    res = [None] * len(tlist)
    curr = 0

    def process_return(q, arg):
        res = callback(arg)
        q.put(res)


    while curr < len(tlist):
        if len(processes) == limit:
            ended = []
            for i, q in enumerate(queues):
                if not q.empty():
                    processes[i].join()
                    ended.append(i)
                    res[indices[i]] = q.get()

            for i in sorted(ended, reverse=True):
                processes.pop(i)
                queues.pop(i)
                indices.pop(i)

            if not ended:
                time.sleep(0.1)
                continue

        queues.append(multiprocessing.Queue())
        indices.append(curr)
        processes.append(multiprocessing.Process(target=process_return, args=(queues[-1], tlist[curr])))
        processes[-1].start()

        curr += 1

    for i, p in enumerate(processes):
        res[indices[i]] = queues[i].get()
        p.join()

    return res