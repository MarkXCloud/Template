import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed, reduce
import template.util as util
from tqdm import tqdm
from collections import OrderedDict


def launch(args):
    module_loader = util.load_module(args.config)

    # load configuration from the module
    saver_config = module_loader.saver_config
    project_config = module_loader.project_config

    saver_config.save_dir = util.generate_config_path(config=args.config, save_dir=saver_config.save_dir)
    project_config.project_dir = saver_config.save_dir

    loggers = [util.SysTracker(logdir=saver_config.save_dir)]
    if args.wandb:
        loggers.append('wandb')
    if args.tb:
        loggers.append('tensorboard')

    accelerator = Accelerator(log_with=loggers,
                              project_config=project_config,
                              gradient_accumulation_steps=args.grad_step)

    set_seed(args.seed, device_specific=True)

    model = module_loader.model
    loss_fn = module_loader.loss_fn
    train_set = module_loader.train_set
    test_set = module_loader.test_set
    optimizer = module_loader.optimizer
    iter_scheduler = module_loader.iter_scheduler
    epoch_scheduler = module_loader.epoch_scheduler
    metric = module_loader.metric

    # basic info for the wandb log
    tracker_config = dict(
        **vars(args),
        model=model.default_cfg,
        loss=loss_fn.name,
        optimizer=dict(name=optimizer.__class__.__name__,
                       **optimizer.defaults),
        iter_scheduler=dict(name=iter_scheduler.__class__.__name__,
                            **iter_scheduler.__dict__),
        epoch_scheduler=dict(name=epoch_scheduler.__class__.__name__,
                             **epoch_scheduler.__dict__),
        learning_rate=module_loader.lr,
        dataset=repr(train_set),
        batch=dict(batch_per_gpu=args.batch_per_gpu,
                   num_proc=accelerator.num_processes,
                   gradient_accumulation_step=args.grad_step,
                   total_batch=args.batch_per_gpu * accelerator.num_processes * args.grad_step),
        save_dir=saver_config.save_dir,
        precision=accelerator.mixed_precision,
        num_proc=accelerator.num_processes
    )

    accelerator.print(util.show_device())
    accelerator.print(f"batch size per gpu = {args.batch_per_gpu}\n"
                      f"num proc = {accelerator.num_processes}\n"
                      f"gradient accumulation step = {args.grad_step}\n"
                      f"so the total batch size = {args.batch_per_gpu * accelerator.num_processes * args.grad_step}")

    accelerator.init_trackers(project_name=module_loader.wandb_log_name,
                              config=tracker_config)

    if args.wandb:
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        wandb_tracker.save(args.config)

    saver = util.Saver(configuration=saver_config)

    accelerator.free_memory()
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_per_gpu, num_workers=args.num_workers,
                              shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_per_gpu, num_workers=args.num_workers,
                             shuffle=False, drop_last=False, pin_memory=True)

    model, optimizer, iter_scheduler, train_loader, test_loader \
        = accelerator.prepare(model, optimizer, iter_scheduler, train_loader, test_loader)
    accelerator.register_for_checkpointing(model, optimizer, iter_scheduler, epoch_scheduler)
    if args.ckpt:
        accelerator.print(f"Resume from: {args.ckpt}")
        accelerator.load_state(args.ckpt)

    if args.compile:
        print("compiling model...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    loss_dict = {}

    del module_loader

    # start training
    for e in range(args.epoch):
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not accelerator.is_local_main_process)

        model.train()
        loss_fn.train()
        for x, y in train_pbar:
            with accelerator.accumulate(model):
                outputs = model(x)
                loss_dict = loss_fn(outputs, y)
                accelerator.backward(loss_dict['loss'])
                optimizer.step()
                optimizer.zero_grad()
                iter_scheduler.step()
        epoch_scheduler.step()

        train_loss_dict = reduce(loss_dict)  # reduce the loss among all devices
        train_loss_dict = OrderedDict({k: v.item() for k, v in train_loss_dict.items()})

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not accelerator.is_local_main_process)

        model.eval()
        loss_fn.eval()
        with torch.no_grad():
            for x, label in test_pbar:
                outputs = model(x)
                metric.add_batch(pred=outputs, label=label)

        metrics = metric.compute()

        trace_log = OrderedDict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.optimizer.param_groups[0]['lr'],
        )

        accelerator.log(trace_log, step=e)

        accelerator.wait_for_everyone()
        if saver.save_latest():
            accelerator.save_state(output_dir=saver_config.save_dir)
        if saver.save_best(metric=metrics):
            accelerator.save(accelerator.get_state_dict(model),
                             f=saver_config.save_dir / "best.pt")

    accelerator.end_training()


def launch_val(args):
    module_loader = util.load_module(args.config)

    accelerator = Accelerator(log_with=util.SysTracker)
    set_seed(args.seed, device_specific=True)
    accelerator.init_trackers(project_name='testing')

    accelerator.free_memory()
    torch.backends.cudnn.benchmark = True

    model = module_loader.model
    test_set = module_loader.test_set
    metric = module_loader.metric

    assert len(test_set) % (args.batch_per_gpu * accelerator.num_processes) == 0, \
        f"test_set with length {len(test_set)} cannot div " \
        f"batch-size({args.batch_size}) * num_gpu({accelerator.num_processes}), which will lead to " \
        f"result in-accurate due to the padding on last batch to align batch-size. Please assign " \
        f"a proper batch-size to make test_set could div."

    model.load_state_dict(torch.load(args.load_from, map_location="cpu"))

    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_per_gpu, num_workers=args.num_workers,
                             shuffle=False, drop_last=False, pin_memory=True)
    model, test_loader = accelerator.prepare(model, test_loader)

    if args.compile:
        print("compiling model...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    test_pbar = tqdm(test_loader, desc=f'Testing', disable=not accelerator.is_local_main_process)

    model.eval()

    with torch.no_grad():
        for x, label in test_pbar:
            outputs = model(x)
            metric.add_batch(pred=outputs, label=label)
    metrics = metric.compute()

    trace_log = OrderedDict(
        **metrics
    )

    accelerator.log(trace_log, step=0)
    accelerator.end_training()
