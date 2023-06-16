import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, reduce
from tqdm import tqdm
from os import makedirs, path as osp

accelerator = Accelerator(log_with=['wandb'])


def launch(args):
    paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver = \
        prepare_everything(args)

    loss_dict = {}

    # start training
    for e in range(epoch):
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not accelerator.is_local_main_process)

        model.train()
        for x, y in train_pbar:
            optimizer.zero_grad()
            loss_dict = paradigm.train(model, x, y, loss)
            accelerator.backward(loss_dict['loss'])
            optimizer.step()
            scheduler.step()

        train_loss_dict = reduce(loss_dict)  # reduce the loss among all devices
        accelerator.print(','.join([f'{k} = {v:.5f}' for k, v in train_loss_dict.items()]))

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not accelerator.is_local_main_process)

        model.eval()
        for x, label in test_pbar:
            pred = paradigm.inference(model, x)
            metric.add_batch(references=label, predictions=pred)

        metrics = metric.compute()
        accelerator.print(','.join([f'{k} = {v:.5f}' for k, v in metrics.items()]))

        saver.save_latest_model(model)
        saver.save_best_model(model, metrics)

        trace_log = dict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.optimizer.param_groups[0]['lr'],
        )
        accelerator.log(trace_log, step=e)

    accelerator.end_training()


def load_module(script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module_script", script_path)
    module_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_script)

    return module_script


def prepare_everything(args):
    set_seed(args.seed)

    module_loader = load_module(args.config)

    # load everything from the module
    paradigm = module_loader.paradigm
    model = module_loader.model
    loss = module_loader.loss
    batch_size = module_loader.batch_size
    train_loader = module_loader.train_loader
    test_loader = module_loader.test_loader
    optimizer = module_loader.optimizer
    scheduler = module_loader.scheduler
    epoch = module_loader.epoch
    metric = module_loader.metric
    saver: Saver = module_loader.saver
    log_name = module_loader.log_name

    # basic info for the wandb log
    tracker_config = dict(
        epoch=epoch,
        model=model.default_cfg['architecture'],
        loss=loss.__class__.__name__,
        optimizer=optimizer.__class__.__name__,
        scheduler=scheduler.__class__.__name__,
        init_learning_rate=optimizer.param_groups[0]['lr'],
        weight_decay=optimizer.param_groups[0]['weight_decay'],
        dataset=train_loader.dataset,
        batch_size=batch_size,
        save_dir=saver.save_dir
    )

    accelerator.init_trackers(log_name, config=tracker_config)

    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    model, optimizer, scheduler, train_loader, test_loader \
        = accelerator.prepare(model, optimizer, scheduler, train_loader, test_loader)

    return paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver


class Saver:
    """
    Saver acts as a scheduler to save the latest model and the best model.
    """

    def __init__(self, save_interval: int, higher_is_better: bool, monitor: str, root: str = './runs/'):
        """
        :param save_interval: when we want to save the latest model, it saves it per $save_step$ epochs.
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        """
        # create save dir
        import inspect, time
        back_filename = inspect.currentframe().f_back.f_code.co_filename  # YOUR/PATH/TO/CONFIG/XXX_cfg.py

        save_dir = osp.join(root, osp.basename(back_filename)[:-3],
                            time.strftime('%Y%m%d_%H_%M_%S', time.localtime(time.time())))
        if accelerator.is_local_main_process:
            import shutil
            makedirs(save_dir)
            shutil.copy(src=back_filename, dst=save_dir)

        self.save_dir = save_dir
        self._save_interval = save_interval
        # count for epochs, when the count meets save_interval, it saves the latest model
        self._cnt = 1

        self.hib = higher_is_better
        self._metric = -1 if higher_is_better else 65535
        self.monitor = monitor

    def save_latest_model(self, model):
        if self._cnt == self._save_interval:
            accelerator.save(accelerator.get_state_dict(model), f=osp.join(self.save_dir, "latest.pt"))
            self._cnt = 1
            accelerator.print(f"Save latest model under {self.save_dir}")
        else:
            self._cnt += 1

    def save_best_model(self, model, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            accelerator.save(accelerator.get_state_dict(model), f=osp.join(self.save_dir, "best.pt"))
            self._metric = metric
            accelerator.print(f"Save new best model under {self.save_dir}")
