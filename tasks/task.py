import torch
import torch.nn
import torch.optim
import framework
import torch.utils.data
import torch.cuda.amp
from typing import Optional, Dict, Any, Tuple, List, Iterable
from interfaces import Result
import optimizer
from interfaces import Result, ModelInterface
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class LastBestMarker:
    iter: int
    loss: float
    accuracy: float


class Task:
    MAX_LENGHT_PER_BATCH = None
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    TRAIN_NUM_WORKERS = 1
    VALID_NUM_WORKERS = 1
    train_set: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    model: torch.nn.Module

    def create_datasets(self):
        raise NotImplementedError()

    def create_model_interface(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @property
    def amp_enabled(self):
        return torch.cuda.is_available() and self.helper.args.amp

    @property
    def time_dim(self) -> int:
        return 1 - self.batch_dim

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.helper.state.best_losses = {}
        self.helper.state.best_accuracies = {}
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()
        self.forward_time_meter = framework.utils.ElapsedTimeMeter()
        self.load_time_meter = framework.utils.ElapsedTimeMeter()
        self.plot_time_meter = framework.utils.ElapsedTimeMeter()

        if self.helper.args.lr_sched.type == "step":
            self.lr_scheduler = optimizer.StepLrSched(self.helper.args.lr, self.helper.args.lr_sched.steps,
                                                      self.helper.args.lr_sched.gamma)

        elif self.helper.args.lr_sched.type == "noam":
            self.lr_scheduler = optimizer.NoamLRSched(self.helper.args.lr, self.helper.args.state_size,
                                                      self.helper.args.lr_warmup)
        else:
            assert False

        self.avg_num_chunks = framework.utils.Average()

        self.create_datasets()
        self.create_loaders()
        self.model = self.create_model()
        self.model = self.model.to(self.helper.device)
        self.create_model_interface()
        self.create_optimizer()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver["scaler"] = self.scaler

        print(f"Total number of model parameters: {sum(p.numel() for p in self.model.parameters())}")

        self.helper.saver["model"] = self.model
        self.helper.restore()

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(vset, batch_size=self.test_batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None) \
                            -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(loader, batch_size=self.helper.args.batch_size,
                                           sampler=framework.loader.sampler.InfiniteSampler(
                                               loader, seed = seed),
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True)

    def set_optimizer_lr(self, lr: float):
        framework.utils.set_lr(self.optimizer, lr)
        if self.helper.state.iter % 100 == 0:
            self.helper.summary.log({"lr": lr})

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def set_lr(self):
        if self.helper.args.lr_sched.type == "step":
            self.set_linear_warmup(self.helper.state.iter, self.helper.args.lr_warmup,
                                   self.lr_scheduler.get(self.helper.state.iter))
        elif self.helper.args.lr_sched.type == "noam":
            self.set_optimizer_lr(self.lr_scheduler.get(self.helper.state.iter))
        else:
            assert False

    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.helper.to_device(data)

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0

            test = set.start_test()
            for d in tqdm(loader):
                d = self.prepare_data(d)
                res = self.model_interface(d)
                digits = self.model_interface.decode_outputs(res)
                loss_sum += res.loss.sum().item() * res.batch_size

                test.step(digits, d)

        self.model.train()
        return test, loss_sum / len(set)

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def update_best_accuracies(self, name: str, accuracy: float, loss: float):
        if name not in self.helper.state.best_losses or loss < self.helper.state.best_losses[name].loss:
                self.helper.state.best_losses[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        if name not in self.helper.state.best_accuracies or accuracy > \
                self.helper.state.best_accuracies[name].accuracy:
            self.helper.state.best_accuracies[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        return {
            f"{name}/time_since_best_loss": self.helper.state.iter - self.helper.state.best_losses[name].iter,
            f"{name}/time_since_best_accuracy": self.helper.state.iter - self.helper.state.best_accuracies[name].iter
        }

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        for name in name_it:
            test, loss = self.validate_on_name(name)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            sum_accuracy += test.accuracy

            charts.update(self.update_best_accuracies(name, test.accuracy, loss))

        charts["mean_accuracy"] = sum_accuracy / len(self.valid_sets)
        charts["mean_loss"] = sum_all_losses / len(self.valid_sets)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            plots["train/loss"] = self.loss_average.get()
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True) * 1000 / 20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True) * 1000 / 20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True) * 1000 / 20

        if self.helper.state.iter % self.helper.args.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        if self.helper.state.iter % 20 == 0:
            plots["average_num_chunks"] = self.avg_num_chunks.get()

        return plots

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def create_optimizer(self):
        if self.helper.args.optimizer == "adam":
            self.set_optimizer(torch.optim.Adam(self.model.parameters(), self.helper.args.lr,
                                                weight_decay=self.helper.args.wd, betas=self.helper.args.adam.betas))
        elif self.helper.args.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.model.parameters(), self.helper.args.lr,
                                               weight_decay=self.helper.args.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.args.optimizer}"

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", self.optimizer, replace=True)

    def get_train_batch(self) -> Dict[str, Any]:
        return next(self.data_iter)

    def run_model(self, data: torch.Tensor) -> Tuple[Result, Dict[str, Any]]:
        res = self.model_interface(data)
        return res, {}

    def chunk_batch_dim(self, data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        if n == 1:
            return [data]

        res = [{} for _ in range(n)]
        for k, v in data.items():
            assert torch.is_tensor(v), "Only tensors are supported by autosplitting"

            bd = self.batch_dim if self.batch_dim < v.ndimension() else 0
            assert v.shape[bd] % n == 0

            for i, c in enumerate(v.chunk(n, dim=bd)):
                res[i][k] = c

        return res

    def is_seq2seq_task(self, data: Dict[str, Any]) -> bool:
        return "in_len" in data and "out_len" in data

    def get_seq_length(self, data: Dict[str, Any]) -> int:
        # This assumes separate encoder and decoder
        return max(data["in"].shape[self.time_dim], data["out"].shape[self.time_dim])

    def get_n_chunks(self, data: Dict[str, Any]) -> int:
        max_length_per_batch = self.helper.args.max_length_per_batch or self.MAX_LENGHT_PER_BATCH
        if self.is_seq2seq_task(data) and max_length_per_batch:
            # The formula below assumes quadratic memory consumption
            return int(2**int(self.get_seq_length(data) / max_length_per_batch))
        return 1

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        plots = {}

        with self.forward_time_meter:
            self.set_lr()
            data = self.prepare_data(self.get_train_batch())
            self.optimizer.zero_grad(set_to_none=True)

            n_chunks = self.get_n_chunks(data)
            res_list = []
            weights = []

            self.avg_num_chunks.add(n_chunks)

            total_out_len = data["out_len"].sum()
            for d in self.chunk_batch_dim(data, n_chunks):
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    res, custom_plots = self.run_model(d)
                    res_list.append(res)
                    plots.update(custom_plots)
                weights.append((d["out_len"].sum() / total_out_len) if "out_len" in d else 1)
                assert torch.isfinite(res_list[-1].loss)
                self.scaler.scale(res_list[-1].loss * weights[-1]).backward()

            self.scaler.unscale_(self.optimizer)
            if self.helper.args.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.helper.state.iter += 1
            res = res_list[0].__class__.merge(res_list, weights)

        return res, plots

    def train(self):
        self.loss_average.reset()

        self.data_iter = iter(self.train_loader)

        while (self.helper.args.stop_after or 10e10) > self.helper.state.iter:
            self.load_time_meter.stop()

            res, plots = self.train_step()
            plots.update(self.plot(res))

            with self.plot_time_meter:
                self.helper.summary.log(plots)

            self.load_time_meter.start()

            self.helper.tick()

    @property
    def test_batch_size(self):
        return self.helper.args.test_batch_size or self.helper.args.batch_size
