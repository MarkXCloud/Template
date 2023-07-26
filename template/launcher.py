import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, reduce
import template.util as util
from collections import OrderedDict


def launch(args):
    console = util.MainConsole(color_system='auto',log_time_format='[%Y.%m.%d %H:%M:%S]')
    with console.status("Loading modules...", spinner="aesthetic", spinner_style='cyan'):
        module_loader = util.load_module(args.config)
    # load configuration from the module
    saver_config = module_loader.saver_config
    project_config = module_loader.project_config

    # generate save_dir with current time
    saver_config.save_dir = util.generate_config_path(config=args.config, save_dir=saver_config.save_dir)
    project_config.project_dir = saver_config.save_dir

    # add loggers
    loggers = [util.SysTracker(logdir=saver_config.save_dir)]
    if args.wandb:
        loggers.append('wandb')
    if args.tb:
        loggers.append('tensorboard')

    accelerator = Accelerator(log_with=loggers,
                              project_config=project_config,
                              gradient_accumulation_steps=args.grad_step)
    # seed all
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
        model=getattr(model, 'default_cfg', model.__class__.__name__),
        loss=loss_fn.name,
        optimizer=dict(name=optimizer.__class__.__name__,
                       **optimizer.defaults),
        iter_scheduler=dict(name=iter_scheduler.__class__.__name__,
                            **iter_scheduler.__dict__),
        epoch_scheduler=dict(name=epoch_scheduler.__class__.__name__,
                             **epoch_scheduler.__dict__),
        learning_rate=module_loader.lr,
        dataset=repr(train_set),
        batch=OrderedDict(batch_per_gpu=args.batch_per_gpu,
                          num_proc=accelerator.num_processes,
                          gradient_accumulation_step=args.grad_step,
                          total_batch_size=args.batch_per_gpu * accelerator.num_processes * args.grad_step),
        save_dir=saver_config.save_dir,
        precision=accelerator.mixed_precision,
        num_proc=accelerator.num_processes
    )
    console.print(tracker_config)

    console.print(util.show_device(), justify="center")
    console.print(util.show_batch_size(tracker_config['batch']), justify="center")

    accelerator.init_trackers(project_name=module_loader.wandb_log_name,
                              config=tracker_config)

    if args.wandb:
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        wandb_tracker.save(args.config)

    saver = util.Saver.from_configuration(config=args.config, configuration=saver_config)

    accelerator.free_memory()
    torch.backends.cudnn.benchmark = True
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_per_gpu,
                                               num_workers=args.num_workers,
                                               collate_fn=train_set.collate_fn, shuffle=True, drop_last=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_per_gpu,
                                              num_workers=args.num_workers,
                                              collate_fn=test_set.collate_fn, shuffle=False, drop_last=False,
                                              pin_memory=True)

    with console.status("Prepare everything...", spinner="aesthetic", spinner_style='cyan'):
        model, optimizer, iter_scheduler, epoch_scheduler, train_loader, test_loader \
            = accelerator.prepare(model, optimizer, iter_scheduler, epoch_scheduler, train_loader, test_loader)
    epoch_scheduler.step_with_optimizer = False

    # resume from checkpoint
    start_epoch = 0
    if args.resume:
        accelerator.load_state(args.resume)
        start_epoch = int(args.resume.split('_')[-1]) + 1
        console.log(f"Resume from: {args.resume}, start epoch: {start_epoch}")

    if args.compile:
        console.log("compile model")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    loss_dict = {}

    del module_loader

    # start training
    console.rule("[bold red]Start Training![/bold red]:fire:", style='#FF3333')
    for e in range(start_epoch, args.epoch):
        model.train()
        loss_fn.train()
        for x, y in util.track(train_loader, description=f'Train epoch {e}', disable=not accelerator.is_local_main_process):
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

        model.eval()
        loss_fn.eval()
        with torch.no_grad():
            for x, label in util.track(test_loader, description=f'Test epoch {e}', disable=not accelerator.is_local_main_process):
                outputs = model(x)
                all_outputs, all_label = accelerator.gather_for_metrics((outputs, label))
                metric.add_batch(pred=all_outputs, label=all_label)

        metrics = metric.compute()

        trace_log = OrderedDict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.optimizer.param_groups[0]['lr'],
        )

        accelerator.log(trace_log, step=e)

        # save latest checkpoint
        if accelerator.is_local_main_process:
            accelerator.save_state()
        # save best model
        if saver.save_best(metric=metrics):
            accelerator.save(accelerator.get_state_dict(model),
                             f=saver_config.save_dir / "best.pt")
        accelerator.wait_for_everyone()
    console.rule("[bold dodger_blue3]Finish Training![/bold dodger_blue3]:ok:",style="cyan")
    accelerator.end_training()


@torch.no_grad()
def launch_val(args):
    console = util.MainConsole(color_system='auto')
    with console.status("Loading modules...", spinner="aesthetic", spinner_style='cyan'):
        module_loader = util.load_module(args.config)
    # load configuration from the module
    saver_config = module_loader.saver_config
    # generate save_dir with current time
    saver_config.save_dir = util.generate_config_path(config=args.config, save_dir=saver_config.save_dir)

    accelerator = Accelerator(log_with=util.SysTracker(logdir=saver_config.save_dir))
    set_seed(args.seed, device_specific=True)
    accelerator.init_trackers(project_name='testing')
    if accelerator.is_local_main_process:
        saver_config.save_dir.mkdir(parents=True, exist_ok=True)

    accelerator.free_memory()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = module_loader.model
    test_set = module_loader.test_set
    metric = module_loader.metric

    del module_loader

    model.load_state_dict(torch.load(args.load_from, map_location="cpu"))

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_per_gpu,
                                              num_workers=args.num_workers,
                                              collate_fn=test_set.collate_fn, shuffle=False, drop_last=False,
                                              pin_memory=True)
    with console.status("Prepare everything...", spinner="aesthetic", spinner_style='cyan'):
        model, test_loader = accelerator.prepare(model, test_loader)

    if args.compile:
        console.log("compile model")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    test_pbar = util.track(test_loader, description=f'Testing', disable=not accelerator.is_local_main_process)

    model.eval()

    for x, label in test_pbar:
        outputs = model(x)
        all_outputs, all_label = accelerator.gather_for_metrics((outputs, label))
        metric.add_batch(pred=all_outputs, label=all_label)
    metrics = metric.compute()

    trace_log = OrderedDict(
        **metrics
    )

    accelerator.log(trace_log, step=0)
    accelerator.end_training()
