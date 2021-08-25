from ..visualize import plot
from ..utils import use_gpu, seed, U
from ..data_structures import DotDict
import framework
import os
import shutil
import sys
from datetime import datetime
import socket
from typing import List, Callable, Optional, Any
from .saver import Saver
from .argument_parser import ArgumentParser
import torch
import time
import subprocess
from copy import deepcopy


def get_plot_config(args):
    assert args.log in ["all", "tb", "wandb"]
    return args.log in ["all", "tb"], args.log in ["all", "wandb"]


class TrainingHelper:
    class Dirs:
        pass

    def __init__(self, register_args: Optional[Callable[[ArgumentParser],None]],
                 wandb_project_name: Optional[str] = None,
                 log_async: bool=False, extra_dirs: List[str]=[]):

        self.is_sweep = False
        self.log_async = log_async
        self.wandb_project_name = wandb_project_name
        self.all_dirs = ["checkpoint", "tensorboard"] + extra_dirs
        self.create_parser()

        if register_args is not None:
            register_args(self.arg_parser)
        self.start()

    def print_env_info(self):
        try:
            import pkg_resources
            print("---------------- Environment information: ----------------")
            installed_packages = pkg_resources.working_set
            print(list(sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])))
            print("----------------------------------------------------------")
        except:
            pass

        try:
            git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL, 
                                stdout=subprocess.PIPE)

            if git.returncode == 0:
                print(f"Git hash: {git.stdout.decode().strip()}")
        except:
            pass

    def create_parser(self):
        self.arg_parser = ArgumentParser(get_train_dir=lambda x: os.path.join("save", x.name) if x.name is not None
                                         else None)
        self.arg_parser.add_argument("-name", type=str, help="Train dir name")
        self.arg_parser.add_argument("-reset", default=False, help="reset training - ignore saves", save=False)
        self.arg_parser.add_argument("-log", default="tb")
        self.arg_parser.add_argument("-save_interval", default="5000", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-wandb_save_interval", default="None", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-seed", default="none", parser=self.arg_parser.int_or_none_parser)
        self.arg_parser.add_argument("-gpu", default="auto", help="use this gpu")
        self.arg_parser.add_argument("-keep_alive", default=False)
        self.arg_parser.add_argument("-sweep_id_for_grid_search", default=0,
                                     help="Doesn't do anything, just to run multiple W&B iterations.")
        self.arg_parser.add_argument("-restore", default="")

    def create_dirs(self):
        self.dirs = self.Dirs()
        self.dirs.base = self.summary.save_dir

        for d in self.all_dirs:
            assert d not in self.dirs.__dict__, f"Directory {d} already exists"
            self.dirs.__dict__[d] = os.path.join(self.dirs.base, d)

        if self.args.reset:
            print("Resetting training state...")
            for d in self.all_dirs:
                shutil.rmtree(self.dirs.__dict__[d], ignore_errors=True)

        for d in self.all_dirs:
            os.makedirs(self.dirs.__dict__[d], exist_ok=True)

    def save_startup_log(self):
        self.arg_parser.save(os.path.join(self.summary.save_dir, "args.json"))
        with open(os.path.join(self.summary.save_dir, "startup_log.txt"), "a+") as f:
            f.write(f"{str(datetime.now())} {socket.gethostname()}: {' '.join(sys.argv)}\n")

    def start_tensorboard(self):
        if self.use_tensorboard:
            os.makedirs(self.dirs.tensorboard, exist_ok=True)
            framework.visualize.tensorboard.start(log_dir=self.dirs.tensorboard)

    def use_cuda(self) -> bool:
        return torch.cuda.is_available() and self.args.gpu.lower() != "none"

    def setup_environment(self):
        use_gpu(self.args.gpu)
        if self.args.seed is not None:
            seed.fix(self.args.seed)

        self.device = torch.device("cuda") if self.use_cuda() else torch.device("cpu")

    def start(self):
        self.args = self.arg_parser.parse_and_try_load()
        self.restore_pending = None

        if self.args.restore:
            # Restore args first such that the rest of the config is loaded correctly. Do not restore the GPU settings.
            assert self.args.reset is False, "Cannot restore and reset at the same time"
            gpu_backup = self.args.gpu
            self.restore_pending = Saver.do_load(self.args.restore)
            self.arg_parser.from_dict(self.restore_pending["run_invariants"]["args"])
            self.args = self.arg_parser.parse_and_try_load()
            self.args.gpu = gpu_backup
            self.args.reset = False

        self.use_tensorboard, self.use_wandb = get_plot_config(self.args)

        constructor = plot.AsyncLogger if self.log_async else plot.Logger

        assert (not self.use_wandb) or (self.wandb_project_name is not None), \
            'Must specify wandb project name if logging to wandb.'

        self.state = DotDict()
        self.state.iter = 0

        assert self.args.name is not None or self.use_wandb, "Either name must be specified or W&B should be used"

        if self.args.restore and self.restore_pending["run_invariants"]["wandb_id"] is not None:
            wandb_args = {
                "project": self.wandb_project_name,
                "id": self.restore_pending["run_invariants"]["wandb_id"]["run_id"],
                "resume": "must"
            }
        else:
            wandb_args = {
                "project": self.wandb_project_name,
                "config": self.arg_parser.to_dict()
            }
        
        self.summary = constructor(save_dir=os.path.join("save", self.args.name) if self.args.name is not None else None,
                                        use_tb=self.use_tensorboard,
                                        use_wandb=self.use_wandb,
                                        wandb_init_args=wandb_args,
                                        wandb_extra_config={
                                            "experiment_name": self.args.name
                                        },
                                        get_global_step = lambda: self.state.iter)

        if self.use_wandb:
            self.print_env_info()

        self.run_invariants = {
            "wandb_id": self.summary.wandb_id,
            "args": self.arg_parser.to_dict()
        }
        self.create_dirs()
        self.save_startup_log()
        self.start_tensorboard()
        self.saver = Saver(self.dirs.checkpoint, self.args.save_interval,
                           keep_every_n_hours=None if self.use_wandb else 4)
        self.saver["state"] = self.state
        self.saver["run_invariants"] = deepcopy(self.run_invariants)
        self.setup_environment()

    def wait_for_termination(self):
        if self.args.keep_alive and self.use_tensorboard and not self.use_wandb:
            print("Done. Waiting for termination")
            while True:
                time.sleep(100)

    def save(self):
        self.saver.save(iter=self.state.iter)
        self.saver.cleanup()

    def tick(self):
        self.saver.tick(iter=self.state.iter)

    def finish(self):
        self.summary.finish()
        if self.is_sweep or self.args.save_interval is None:
            self.save()

        self.wait_for_termination()

    def to_device(self, data: Any) -> Any:
        return U.apply_to_tensors(data, lambda d: d.to(self.device))

    def restore(self):
        if self.restore_pending is not None:
            assert self.saver.load_data(self.restore_pending), "Restoring failed."
            self.restore_pending = None
            restored = True
        else:
            restored = self.saver.load()

        if restored:
            # Do not restore these things
            self.saver.register("run_invariants", deepcopy(self.run_invariants), replace=True)

    def get_storage_path(self, path: str) -> str:
        path = os.path.join(self.dirs.export, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
