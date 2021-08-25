import torch
import os
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from ..utils import U
from typing import Dict, Tuple, List, Optional, Callable, Union
import threading
import atexit
from torch.multiprocessing import Process, Queue, Event
from queue import Empty as EmptyQueue
import sys
import itertools
import PIL

wandb = None
plt = None
make_axes_locatable = None


def import_matplotlib():
    global plt
    global make_axes_locatable
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable


class CustomPlot:
    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        pass

    def to_wandb(self):
        return None


class Histogram(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], n_bins: int = 64):
        if torch.is_tensor(data):
            data = data.detach().cpu()

        self.data = data
        self.n_bins = n_bins

    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        summary_writer.add_histogram(name, self.data, global_step, max_bins=self.n_bins)

    def to_wandb(self):
        return wandb.Histogram(self.data, num_bins=self.n_bins)


class Image(CustomPlot):
    def __init__(self, data: Union[torch.Tensor, np.ndarray], caption: Optional[str] = None):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        self.data = data.astype(np.float32)
        self.caption = caption

    def to_tensorboard(self, name, summary_writer, global_step):
        if self.data.shape[-1] in [1,3]:
            data = np.transpose(self.data, (2,0,1))
        else:
            data = self.data
        summary_writer.add_image(name, data, global_step)

    def to_wandb(self):
        if self.data.shape[0] in [1, 3]:
            data = np.transpose(self.data, (1,2,0))
        else:
            data = self.data

        data = PIL.Image.fromarray(np.ascontiguousarray((data*255.0).astype(np.uint8)), mode="RGB")
        return wandb.Image(data, caption = self.caption)


class Scalars(CustomPlot):
    def __init__(self, scalar_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]]):
        self.values = {k: v.item() if torch.is_tensor(v) else v for k, v in scalar_dict.items()}
        self.leged = sorted(self.values.keys())

    def to_tensorboard(self, name, summary_writer, global_step):
        v = {k: v for k, v in self.values.items() if v == v}
        summary_writer.add_scalars(name, v, global_step)

    def to_wandb(self):
        return self.values


class Scalar(CustomPlot):
    def __init__(self, val: Union[torch.Tensor, np.ndarray, int, float]):
        if torch.is_tensor(val):
            val = val.item()

        self.val = val

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_scalar(name, self.val, global_step)

    def to_wandb(self):
        return self.val


class XYChart(CustomPlot):
    def __init__(self, data: Dict[str, List[Tuple[float, float]]], markers: List[Tuple[float,float]] = [],
                 xlim = (None, None), ylim = (None, None)):
        import_matplotlib()

        self.data = data
        self.xlim = xlim
        self.ylim = ylim
        self.markers = markers

    def matplotlib_plot(self):
        f = plt.figure()
        names = list(sorted(self.data.keys()))

        for n in names:
            plt.plot([p[0] for p in self.data[n]], [p[1] for p in self.data[n]])

        if self.markers:
            plt.plot([a[0] for a in self.markers], [a[1] for a in self.markers], linestyle='', marker='o',
                 markersize=2, zorder=999999)

        plt.legend(names)
        plt.ylim(*self.xlim)
        plt.xlim(*self.ylim)

        return f

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.matplotlib_plot(), global_step)

    def to_wandb(self):
        return self.matplotlib_plot()



class Heatmap(CustomPlot):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], xlabel: str, ylabel: str,
                 round_decimals: Optional[int] = None, x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.round_decimals = round_decimals
        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks

    def to_matplotlib(self):
        figure, ax = plt.subplots(figsize=(self.map.shape[0]*0.25 + 2, self.map.shape[1]*0.15+1.5))
        im = plt.imshow(self.map, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')

        x_marks = self.x_marks if self.x_marks is not None else [str(i) for i in range(self.map.shape[1])]
        assert len(x_marks) == self.map.shape[1]

        y_marks = self.y_marks if self.y_marks is not None else [str(i) for i in range(self.map.shape[0])]
        assert len(y_marks) == self.map.shape[0]

        plt.xticks(np.arange(self.map.shape[1]), x_marks, rotation=45, fontsize=8, ha="right", rotation_mode="anchor")
        plt.yticks(np.arange(self.map.shape[0]), y_marks, fontsize=8)

        # Use white text if squares are dark; otherwise black.
        threshold = self.map.max() / 2.

        rmap = np.around(self.map, decimals=self.round_decimals) if self.round_decimals is not None else self.map
        for i, j in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            color = "white" if self.map[i, j] > threshold else "black"
            plt.text(j, i, rmap[i, j], ha="center", va="center", color=color, fontsize=8)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.25, pad=0.1)
        plt.colorbar(im, cax)

        plt.tight_layout()
        return figure

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_figure(name, self.to_matplotlib(), global_step)

    def to_wandb(self):
        return wandb.Image(self.to_matplotlib())


class ConfusionMatrix(Heatmap):
    def __init__(self, map: Union[torch.Tensor, np.ndarray], class_names: Optional[List[str]] = None,
                 x_marks: Optional[List[str]] = None, y_marks: Optional[List[str]] = None):

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        map = np.transpose(map, (1, 0))
        map = map.astype('float') / map.sum(axis=1).clip(1e-6, None)[:, np.newaxis]

        if class_names is not None:
            assert x_marks is None and y_marks is None
            x_marks = y_marks = class_names

        super().__init__(map, "predicted", "real", round_decimals=2, x_marks = x_marks, y_marks = y_marks)


class TextTable(CustomPlot):
    def __init__(self, header: List[str], data: List[List[str]]):
        self.header = header
        self.data = data

    def to_markdown(self):
        res = " | ".join(self.header)+"\n"
        res += " | ".join("---" for _ in self.header)+"\n"
        return res+"\n".join([" | ".join([x.replace("|", "&#124;") for x in l]) for l in self.data])

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_text(name, self.to_markdown(), global_step)

    def to_wandb(self):
        return wandb.Table(data=self.data, columns=self.header)


class PlotAsync:
    @staticmethod
    def worker(self, fn, *args):
        try:
            self.result = fn(*args)
        except:
            self.failed = True
            raise

    def __init__(self, fn: Callable[[any], Dict[str, any]], args: Tuple=()):
        self.result = None
        self.failed = False

        args = U.apply_to_tensors(args, lambda x: x.detach().cpu().clone())

        self.thread = threading.Thread(target = self.worker, args=(self, fn, *args))
        self.thread.start()

    def get(self, wait: bool) -> Optional[Dict[str, any]]:
        if (self.result is None and not wait) or self.failed:
            return None

        self.thread.join()
        return self.result


class Logger:
    @staticmethod
    def parse_switch_string(s: str) -> Tuple[bool,bool]:
        s = s.lower()
        if s=="all":
            return True, True
        elif s=="none":
            return False, False

        use_tb, use_wandb =  False, False
        s = s.split(",")
        for p in s:
            if p=="tb":
                use_tb = True
            elif p=="wandb":
                use_wandb = True
            else:
                assert False, "Invalid visualization switch: %s" % p

        return use_tb, use_wandb

    def create_loggers(self):
        self.is_sweep = False
        self.wandb_id = {}
        global wandb

        if self.use_wandb:
            import wandb
            wandb.init(**self.wandb_init_args)
            self.wandb_id = {
                "sweep_id": wandb.run.sweep_id,
                "run_id": wandb.run.id
            }
            self.is_sweep = bool(wandb.run.sweep_id)
            wandb.config["is_sweep"] = self.is_sweep
            wandb.config.update(self.wandb_extra_config)

            self.save_dir = os.path.join(wandb.run.dir)

        os.makedirs(self.save_dir, exist_ok=True)
        self.tb_logdir = os.path.join(self.save_dir, "tensorboard")

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tb_logdir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=self.tb_logdir, flush_secs=30)
        else:
            self.summary_writer = None

    def __init__(self, save_dir: Optional[str] = None, use_tb: bool = False, use_wandb: bool = False,
                 get_global_step: Optional[Callable[[], int]] = None, wandb_init_args={}, wandb_extra_config={}):
        global plt
        global wandb

        import_matplotlib()

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.save_dir = save_dir
        self.get_global_step = get_global_step
        self.wandb_init_args = wandb_init_args
        self.wandb_extra_config = wandb_extra_config

        self.create_loggers()

    def flatten_dict(self, dict_of_elems: Dict) -> Dict:
        res = {}
        for k, v in dict_of_elems.items():
            if isinstance(v, dict):
                v = self.flatten_dict(v)
                for k2, v2 in v.items():
                    res[k+"/"+k2] = v2
            else:
                res[k] = v
        return res

    def get_step(self, step: Optional[int] = None) -> Optional[int]:
        if step is None and self.get_global_step is not None:
            step = self.get_global_step()

        return step

    def log(self, plotlist: Union[List, Dict, PlotAsync], step: Optional[int] = None):
        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p.get(True) if isinstance(p, PlotAsync) else p for p in plotlist if p]
        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        d = {}
        for p in plotlist:
            d.update(p)

        self.log_dict(d, step)

    def log_dict(self, dict_of_elems: Dict, step: Optional[int] = None):
        dict_of_elems = self.flatten_dict(dict_of_elems)

        if not dict_of_elems:
            return

        dict_of_elems = {k: v.item() if torch.is_tensor(v) and v.nelement()==1 else v for k, v in dict_of_elems.items()}
        dict_of_elems = {k: Scalar(v) if isinstance(v, (int, float)) else v for k, v in dict_of_elems.items()}

        step = self.get_step(step)

        if self.use_wandb:
            wandbdict = {}
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v = v.to_wandb()
                    if v is None:
                        continue

                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            wandbdict[k+"/"+k2] = v2
                    else:
                        wandbdict[k] = v
                elif isinstance(v, plt.Figure):
                    wandbdict[k] = v
                else:
                    assert False, f"Invalid data type {type(v)}"

            wandbdict["iteration"] = step
            wandb.log(wandbdict)

        if self.summary_writer is not None:
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomPlot):
                    v.to_tensorboard(k, self.summary_writer, step)
                elif isinstance(v, plt.Figure):
                    self.summary_writer.add_figure(k, v, step)
                else:
                    assert False, f"Unsupported type {type(v)} for entry {k}"

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def flush(self):
        pass

    def finish(self):
        pass


class AsyncLogger(Logger):
    @staticmethod
    def log_fn(self, stop_event: Event):
        try:
            self._super_create_loggers()
            self.resposne_queue.put({k: self.__dict__[k] for k in ["save_dir", "tb_logdir", "is_sweep"]})

            while True:
                try:
                    cmd = self.draw_queue.get(True, 0.1)
                except EmptyQueue:
                    if stop_event.is_set():
                        break
                    else:
                        continue

                self._super_log(*cmd)
                self.resposne_queue.put(True)
        except:
            print("Logger process crashed.")
            raise
        finally:
            print("Logger: syncing")
            if self.use_wandb:
                wandb.join()

            stop_event.set()
            print("Logger process terminating...")

    def create_loggers(self):
        self._super_create_loggers = super().create_loggers
        self.stop_event = Event()
        self.proc = Process(target=self.log_fn, args=(self, self.stop_event))
        self.proc.start()

        atexit.register(self.finish)

    def __init__(self, *args, **kwargs):
        self.queue = []

        self.draw_queue = Queue()
        self.resposne_queue = Queue()
        self._super_log = super().log
        self.waiting = 0

        super().__init__(*args, **kwargs)

        self.__dict__.update(self.resposne_queue.get(True))

    def log(self, plotlist, step=None):
        if self.stop_event.is_set():
            return

        if not isinstance(plotlist, list):
            plotlist = [plotlist]

        plotlist = [p for p in plotlist if p]
        if not plotlist:
            return

        plotlist = U.apply_to_tensors(plotlist, lambda x: x.detach().cpu())
        self.queue.append((plotlist, step))
        self.flush(wait = False)

    def enqueue(self, data, step: Optional[int]):
        self.draw_queue.put((data, step))
        self.waiting += 1

    def wait_logger(self, wait = False):
        cond = (lambda: not self.resposne_queue.empty()) if not wait else (lambda: self.waiting>0)
        already_printed = False
        while cond() and not self.stop_event.is_set():
            will_wait = self.resposne_queue.empty()
            if will_wait and not already_printed:
                already_printed = True
                sys.stdout.write("Warning: waiting for logger... ")
                sys.stdout.flush()
            try:
                self.resposne_queue.get(True, 0.2)
            except EmptyQueue:
                continue
            self.waiting -= 1

        if already_printed:
            print("done.")

    def flush(self, wait: bool = True):
        while self.queue:
            plotlist, step = self.queue[0]

            for i, p in enumerate(plotlist):
                if isinstance(p, PlotAsync):
                    res = p.get(wait)
                    if res is not None:
                        plotlist[i] = res
                    else:
                        if wait:
                            assert p.failed
                            # Exception in the worker thread
                            print("Exception detected in a PlotAsync object. Syncing logger and ignoring further plots.")
                            self.wait_logger(True)
                            self.stop_event.set()
                            self.proc.join()

                        return

            self.queue.pop(0)
            self.enqueue(plotlist, step)

        self.wait_logger(wait)

    def finish(self):
        if self.stop_event.is_set():
            return

        self.flush(True)
        self.stop_event.set()
        self.proc.join()
