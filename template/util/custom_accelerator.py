import shutil
import re
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from pathlib import Path
from dataclasses import dataclass,asdict
from types import MethodType
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
    is_tpu_available,
    is_xpu_available,
    is_npu_available,
    save_fsdp_model,
    save_fsdp_optimizer,
    is_deepspeed_available,
    DistributedType,
    ProjectConfiguration,
    get_mixed_precision_context_manager,
    convert_outputs_to_fp32,
    has_transformer_engine_layers,
    convert_model,
    is_fp8_available,
    DynamoBackend,
    is_torch_version

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

if is_fp8_available():
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast
from typing import Union, List, Any, Callable
from template.util.rich import MainConsole

if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

console = MainConsole(color_system='auto', log_time_format='[%Y.%m.%d %H:%M:%S]')


@dataclass
class SaverConfiguration:
    automatic_checkpoint_naming: bool
    total_limit: int
    higher_is_better: bool
    monitor: str
    project_dir: Union[Path | str] = Path('')

    def generate_config_path(self, config: str):
        """generate save direction by $CONFIG$/$current time$"""
        self.project_dir = Path(self.project_dir) / Path(config).stem / datetime.datetime.today().strftime(
            '%Y%m%d_%H_%M_%S')
    def to_dict(self):
        return {k:str(v) for k,v in asdict(self).items()}



class SimplerAccelerator(Accelerator):
    def __init__(self, *args, config: str, saver_config: SaverConfiguration | None = None, **kwargs):
        project_config = ProjectConfiguration(project_dir=saver_config.project_dir,
                                              automatic_checkpoint_naming=saver_config.automatic_checkpoint_naming,
                                              total_limit=saver_config.total_limit) if saver_config else None
        super(SimplerAccelerator, self).__init__(*args, **kwargs, project_config=project_config)

        saver_config.project_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Current project save dir: {saver_config.project_dir}")
        shutil.copy(src=config, dst=saver_config.project_dir)

        self.hib = saver_config.higher_is_better
        self.monitor = saver_config.monitor
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

    def prepare_model(self, model: torch.nn.Module, device_placement: bool = None, evaluation_mode: bool = False):
        """
        Prepares a PyTorch model for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model to prepare. You don't need to prepare a model if it is used only for inference without
                any kind of mixed precision
            device_placement (`bool`, *optional*):
                Whether or not to place the model on the proper device. Will default to `self.device_placement`.
            evaluation_mode (`bool`, *optional*, defaults to `False`):
                Whether or not to set the model for evaluation only, by just applying mixed precision and
                `torch.compile` (if configured in the `Accelerator` object).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model is defined
        >>> model = accelerator.prepare_model(model)
        ```
        """
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
        self._models.append(model)
        # We check only for models loaded with `accelerate`
        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = False
        for m in model.modules():
            if hasattr(m, "hf_device_map"):
                has_hf_device_map = True
                break

        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(
                model, "hf_device_map", False
        ):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1 and self.distributed_type != DistributedType.NO:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices in any distributed mode."
                    " In order to use 8-bit models that have been loaded across multiple GPUs the solution is to use Naive Pipeline Parallelism."
                    " Therefore you should not specify that you are under any distributed regime in your accelerate config."
                )
            current_device = list(model_devices)[0]
            current_device_index = current_device.index if isinstance(current_device, torch.device) else current_device

            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not has_hf_device_map:
            model = model.to(self.device)

        if self.native_amp:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
            if self.mixed_precision == "fp16":
                if is_npu_available():
                    new_forward = torch.npu.amp.autocast(dtype=torch.float16)(model_forward_func)
                else:
                    new_forward = torch.cuda.amp.autocast(dtype=torch.float16)(model_forward_func)
            elif self.mixed_precision == "bf16" and self.distributed_type != DistributedType.TPU:
                new_forward = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(model_forward_func)

            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
            else:
                model.forward = convert_outputs_to_fp32(new_forward)
        elif self.mixed_precision == "fp8":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            cuda_device_capacity = torch.cuda.get_device_capability()
            fp8_enabled = cuda_device_capacity[0] >= 9 or (
                    cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
            )
            if not fp8_enabled:
                console.warn(
                    f"The current device has compute capability of {cuda_device_capacity} which is "
                    "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                    "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                )
            model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if not evaluation_mode:
            if self.distributed_type in (
                    DistributedType.MULTI_GPU,
                    DistributedType.MULTI_NPU,
                    DistributedType.MULTI_XPU,
            ):
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
                    )
            elif self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                if type(model) != FSDP:
                    self.state.fsdp_plugin.set_auto_wrap_policy(model)
                    fsdp_plugin = self.state.fsdp_plugin
                    kwargs = {
                        "sharding_strategy": fsdp_plugin.sharding_strategy,
                        "cpu_offload": fsdp_plugin.cpu_offload,
                        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                        "mixed_precision": fsdp_plugin.mixed_precision_policy,
                        "sync_module_states": fsdp_plugin.sync_module_states,
                        "backward_prefetch": fsdp_plugin.backward_prefetch,
                        "forward_prefetch": fsdp_plugin.forward_prefetch,
                        "use_orig_params": fsdp_plugin.use_orig_params,
                        "param_init_fn": fsdp_plugin.param_init_fn,
                        "ignored_modules": fsdp_plugin.ignored_modules,
                        "ignored_parameters": fsdp_plugin.ignored_parameters,
                        "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                        "device_id": self.device,
                    }
                    model = FSDP(model, **kwargs)
                self._models[-1] = model
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
            elif self.distributed_type == DistributedType.TPU and self.state.fork_launched:
                model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            # commands: --dynamo_backend inductor --dynamo_mode default/reduce-overhead/max-autotune
            console.print("use compiled model: ", self.state.dynamo_plugin.to_dict())
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model, **self.state.dynamo_plugin.to_dict())
        return model

    @on_main_process
    def save_state(self, output_dir: Union[Path | str] = Path(''), **save_model_func_kwargs):
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = self.project_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [output_dir / folder for folder in output_dir.iterdir()]
            if self.project_configuration.total_limit is not None and (
                    len(folders) + 1 > self.project_configuration.total_limit
            ):

                def _inner(folder: Path):
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
    def save_best_model(self, metric, model):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            self._metric = metric

            # get the state_dict of the model
            state_dict = self.get_state_dict(model, unwrap=True)

            console.log(f"Save new [bold cyan]best model[/bold cyan] under {self.BEST_WEIGHTS}")
            torch.save(state_dict, f=self.BEST_WEIGHTS)


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
