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
from accelerate.state import PartialState
from accelerate.utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    get_pretty_name,
    is_xpu_available,
    save_fsdp_model,
    save_fsdp_optimizer,
    is_deepspeed_available,
    DistributedType,
    ProjectConfiguration,
)

if is_deepspeed_available():
    import deepspeed

    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )
from typing import Union, List,Any,Callable
from template.util.rich import MainConsole

console = MainConsole(color_system='auto', log_time_format='[%Y.%m.%d %H:%M:%S]')


@dataclass
class SaverConfiguration:
    automatic_checkpoint_naming: bool
    total_limit: int
    higher_is_better: bool
    monitor: str
    project_dir: Union[Path | str] = Path('')


class SimplerAccelerator(Accelerator):
    def __init__(self, *args, config:str,saver_config: SaverConfiguration | None = None, **kwargs):
        project_config = ProjectConfiguration(project_dir=saver_config.project_dir,
                                              automatic_checkpoint_naming=saver_config.automatic_checkpoint_naming,
                                              total_limit=saver_config.total_limit)
        super(SimplerAccelerator,self).__init__(*args, **kwargs, project_config=project_config)
        self.hib = saver_config.higher_is_better
        self.monitor = saver_config.monitor
        saver_config.project_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Current project save dir: {saver_config.project_dir}")
        shutil.copy(src=config, dst=saver_config.project_dir)
        self._metric = -1 if self.hib else 65535
        self.BEST_WEIGHTS = saver_config.project_dir / "best.pt"

    def on_main_process(self, function: Callable[..., Any] = None):
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_main_process(function)(*args, **kwargs)

        return _inner

    @on_main_process
    def save_state(self, output_dir: str = None, **save_model_func_kwargs):
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = self.project_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [output_dir / folder for folder in output_dir.iterdir()]
            if self.project_configuration.total_limit is not None and (
                    len(folders) + 1 > self.project_configuration.total_limit
            ):

                def _inner(folder:Path):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder.name)))[0]

                folders.sort(key=_inner)
                console.log(
                    f"Deleting {len(folders) + 1 - self.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - self.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = output_dir / f"checkpoint_{self.save_iteration}"
            if output_dir.exists():
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({self.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Saving current state to {output_dir}")

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                console.log("Saving FSDP model")
                save_fsdp_model(self.state.fsdp_plugin, self, model, output_dir, i)
                console.log(f"FSDP Model saved to output dir {output_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                console.log("Saving DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.save_checkpoint(output_dir, ckpt_id, **save_model_func_kwargs)
                console.log(f"DeepSpeed Model and Optimizer saved to output dir {output_dir / ckpt_id}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                console.log("Saving Megatron-LM Model, Optimizer and Scheduler")
                model.save_checkpoint(output_dir)
                console.log(f"Megatron-LM Model , Optimizer and Scheduler saved to output dir {output_dir}")
            else:
                weights.append(self.get_state_dict(model, unwrap=False))

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for opt in self._optimizers:
                console.log("Saving FSDP Optimizer")
                save_fsdp_optimizer(self.state.fsdp_plugin, self, opt, self._models[i], output_dir, i)
                console.log(f"FSDP Optimizer saved to output dir {output_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir, weights, optimizers, schedulers, self.state.process_index, self.scaler
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i)
        self.project_configuration.iteration += 1
        return save_location

    @on_main_process
    def save_best_model(self,metric,model):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            self._metric = metric

            # get the state_dict of the model
            state_dict = self.get_state_dict(model,unwrap=True)

            console.log(f"Save new [bold cyan]best model[/bold cyan] under {self.BEST_WEIGHTS}")
            torch.save(state_dict, f=self.BEST_WEIGHTS)



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
