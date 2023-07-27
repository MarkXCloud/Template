import shutil
import re
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
from dataclasses import dataclass
import datetime
from accelerate import Accelerator
from accelerate.tracking import on_main_process
from accelerate.utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    get_pretty_name,
    is_xpu_available,
)
from typing import Union, List
from template.util.rich import MainConsole

console = MainConsole(color_system='auto', log_time_format='[%Y.%m.%d %H:%M:%S]')


@dataclass
class SaverConfiguration:
    higher_is_better: bool
    monitor: str
    save_dir: Union[Path | str] = Path('')


class Saver:
    """
    Saver can save the latest model and the best model.
    """
    main_process_only = True

    @on_main_process
    def __init__(self, higher_is_better: bool, monitor: str, save_dir: Union[Path | str], config: str):
        """
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        :param save_dir: save direction, generate by generate_config_path()
        """

        self.hib = higher_is_better
        self.monitor = monitor

        save_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Current project save dir: {save_dir}")
        shutil.copy(src=config, dst=save_dir)
        self._metric = -1 if self.hib else 65535
        self.BEST_WEIGHTS = save_dir / "best.pt"

    @on_main_process
    def save_best(self, metric, accelerator: Accelerator, model):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            self._metric = metric
            console.log(f"Save new [bold cyan]best model[/bold cyan] under {self.BEST_WEIGHTS}")
            torch.save(accelerator.get_state_dict(model), f=self.BEST_WEIGHTS)

    @on_main_process
    def save_state(self, accelerator: Accelerator, output_dir: Path = None):
        if accelerator.project_configuration.automatic_checkpoint_naming:
            output_dir = accelerator.project_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        if accelerator.project_configuration.automatic_checkpoint_naming:
            folders = [output_dir / folder for folder in output_dir.iterdir()]
            if accelerator.project_configuration.total_limit is not None and (
                    len(folders) + 1 > accelerator.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

                folders.sort(key=_inner)
                console.log(
                    f"Deleting {len(folders) + 1 - accelerator.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - accelerator.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = output_dir / f"checkpoint_{accelerator.save_iteration}"
            if output_dir.exists():
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({accelerator.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Saving current state to {output_dir}")

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = [accelerator.get_state_dict(model, unwrap=False) for model in accelerator._models]

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in accelerator._save_model_state_pre_hook.values():
            hook(accelerator._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir, weights, accelerator._optimizers, accelerator._schedulers, accelerator.state.process_index,
            accelerator.scaler
        )
        for i, obj in enumerate(accelerator._custom_objects):
            save_custom_state(obj, output_dir, i)
        accelerator.project_configuration.iteration += 1
        return save_location

    @classmethod
    def from_configuration(cls, config: str, configuration: SaverConfiguration):
        save_dir = configuration.save_dir
        hib = configuration.higher_is_better
        monitor = configuration.monitor
        return cls(save_dir=save_dir, monitor=monitor, higher_is_better=hib, config=config)

    @property
    def best_metric(self):
        return self._metric

    def __repr__(self):
        return f"save directory:{self.save_dir}\n" \
               f"monitor: {self.monitor}\n" \
               f"high is better: {self.hib}\n" \
               f"current best metric: {self.best_metric}"


def generate_config_path(config: str, save_dir: str):
    """generate save direction by $CONFIG$/$current time$"""
    save_dir = Path(save_dir) / Path(config).stem / datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S')

    return save_dir


def save_accelerator_state(
        output_dir: Path,
        model_states: List[dict],
        optimizers: list,
        schedulers: list,
        process_index: int,
        scaler: GradScaler = None,
):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.
    most copy from accelerate

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (`List[torch.nn.Module]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save
    """
    # Model states
    for i, state in enumerate(model_states):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        output_model_file = output_dir / weights_name
        torch.save(state, output_model_file)
        console.log(f"Model weights saved in {output_model_file}")
    # Optimizer states
    for i, opt in enumerate(optimizers):
        state = opt.state_dict()
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        output_optimizer_file = output_dir / optimizer_name
        torch.save(state, output_optimizer_file)
        console.log(f"Optimizer state saved in {output_optimizer_file}")
    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        state = scheduler.state_dict()
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        output_scheduler_file = output_dir / scheduler_name
        torch.save(state, output_scheduler_file)
        console.log(f"Scheduler state saved in {output_scheduler_file}")
    # GradScaler state
    if scaler is not None:
        state = scaler.state_dict()
        output_scaler_file = output_dir / SCALER_NAME
        torch.save(state, output_scaler_file)
        console.log(f"Gradient scaler state saved in {output_scaler_file}")
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    if is_xpu_available():
        states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()
    else:
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()

    output_states_file = output_dir / states_name
    torch.save(states, output_states_file)
    console.log(f"Random states saved in {output_states_file}")
    return output_dir


def save_custom_state(obj, path, index: int = 0):
    """
    Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
    """
    # Should this be the right way to get a qual_name type value from `obj`?
    save_location = Path(path) / f"custom_checkpoint_{index}.pkl"
    console.log(f"Saving the state of {get_pretty_name(obj)} to {save_location}")
    torch.save(obj.state_dict(), save_location)
