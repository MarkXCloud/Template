import importlib
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, reduce
from tqdm import tqdm
import os
import time

accelerator = Accelerator(log_with=['wandb'], project_dir='./runs/')


def launch(config):
    paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver = \
        prepare_everything(config)

    loss_dict = {}
    for e in range(epoch):
        train_pbar = tqdm(train_loader, desc=f'Train epoch {e}', disable=not accelerator.is_local_main_process)

        model.train()
        for x, y in train_pbar:
            optimizer.zero_grad()
            loss_dict = paradigm.train(model, x, y, loss)
            accelerator.backward(loss_dict['loss'])
            optimizer.step()

        train_loss_dict = reduce(loss_dict)
        accelerator.print(','.join([f'{k} = {v}' for k, v in loss_dict.items()]))

        test_pbar = tqdm(test_loader, desc=f'Test epoch {e}', disable=not accelerator.is_local_main_process)

        model.eval()
        for x, label in test_pbar:
            pred = paradigm.inference(model, x)
            metric.add_batch(references=label, predictions=pred)

        metrics = metric.compute()
        accelerator.print(','.join([f'{k} = {v}' for k, v in metrics.items()]))

        saver.save_latest_model(model)
        saver.save_best_model(model, metrics)

        trace_log = dict(
            **metrics,
            **train_loss_dict,
            learning_rate=optimizer.__dict__['optimizer'].__dict__['param_groups'][0]['lr'],
        )
        accelerator.log(trace_log, step=e)

        scheduler.step()
    accelerator.end_training()


def prepare_everything(config):

    set_seed(config.seed)

    pyfile = config.cfg
    assert pyfile.endswith('.py'), "cfg file should be a .py file"
    module = pyfile.replace('/', '.')[:-3]
    module_loader = importlib.import_module(module)

    from utils.Paradigm import BaseParadigm
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer

    paradigm: BaseParadigm = module_loader.paradigm
    model: Module = module_loader.model
    loss: Module = module_loader.loss
    input_size: tuple = module_loader.input_size
    train_loader: DataLoader = module_loader.train_loader
    test_loader: DataLoader = module_loader.test_loader
    optimizer: Optimizer = module_loader.optimizer
    scheduler = module_loader.scheduler
    epoch: int = module_loader.epoch
    metric = module_loader.metric
    saver: Saver = module_loader.saver
    log_name: str = module_loader.log_name

    # basic info for the wandb log
    tracker_config = dict(
        epoch=epoch,
        model=model.__dict__['default_cfg']['architecture'],
        loss=loss.__class__.__name__,
        optimizer=optimizer.__class__.__name__,
        scheduler=scheduler.__class__.__name__,
        init_learning_rate=optimizer.__dict__['param_groups'][0]['lr'],
        weight_decay=optimizer.__dict__['param_groups'][0]['weight_decay'],
        dataset=train_loader.__class__.__name__,
        batch_size=input_size[0],
        save_dir=saver.save_dir
    )

    accelerator.init_trackers(log_name, config=tracker_config)

    torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    device = accelerator.device
    model.to(device)

    things = [model, optimizer, scheduler, train_loader, test_loader]
    model, optimizer, scheduler, train_loader, test_loader \
        = accelerator.prepare(*things)

    return paradigm, model, loss, train_loader, test_loader, optimizer, scheduler, epoch, metric, saver


class Saver:
    """
    Saver acts as a scheduler to save the latest model and the best model.
    """

    def __init__(self, save_interval: int, higher_is_better: bool, monitor: str):
        """
        :param save_interval: when we want to save the latest model, it saves it per $save_step$ epochs.
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        """
        # create save dir
        save_dir = './runs/' + time.strftime('%Y%m%d_%H_%M_%S', time.gmtime(time.time()))
        if accelerator.is_local_main_process:
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self._save_interval = save_interval
        # count for epochs, when the count meets save_interval, it saves the latest model
        self._cnt = 0

        self.hib = higher_is_better
        self._metric = -1 if higher_is_better else 65535
        self.monitor = monitor

    def save_latest_model(self, model):
        if self._cnt == self._save_interval:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self.save_dir, "latest.pt"))
            self._cnt = 0
            accelerator.print(f"Save latest model under {self.save_dir}")
        else:
            self._cnt += 1

    def save_best_model(self, model, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self.save_dir, "best.pt"))
            self._metric = metric
            accelerator.print(f"Save new best model under {self.save_dir}")

# class HyperSearcher:
#     def __init__(self, config):
#         cfg = config.cfg
#         assert cfg.endswith('.py'), "cfg file should be a .py file"
#         module = cfg.replace('/', '.')[:-3]
#         module_loader = importlib.import_module(module)
#
#         self.model = module_loader.model
#         self.loss = module_loader.loss
#         self.train_set = module_loader.train_set
#         self.test_set = module_loader.test_set
#         self.epoch = module_loader.epoch
#         assert self.epoch > 0
#         self.optimizer, self.train_loader, self.test_loader = None, None, None
#
#         device = accelerator.device
#         self.model.to(device)
#
#     def init_seeds(self, seed):
#         # accelerate.set_seed() can seed all
#         torch.cuda.empty_cache()
#         set_seed(seed)
#         torch.backends.cudnn.benchmark = True
#
#     def hyp_search(self, trial):
#         batch_size = trial.suggest_int('batch_size', 16, 64)
#         lr = trial.suggest_float('lr', 1e-4, 1e-2, step=0.0001)
#         self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
#         self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=4)
#         self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=4)
#
#         self.model, self.optimizer, self.train_loader, self.test_loader \
#             = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)
#         total_loss = 0
#         for e in range(self.epoch):
#             train_pbar = tqdm(self.train_loader, disable=not accelerator.is_local_main_process)
#             train_pbar.set_description(f'Train epoch {e}')
#
#             self.model.train()
#             for x, y in train_pbar:
#                 self.optimizer.zero_grad()
#                 pred = self.model(x)
#                 loss = self.loss(pred, y)
#
#                 accelerator.backward(loss)
#                 self.optimizer.step()
#
#                 train_pbar.set_postfix(loss=loss.cpu().item())
#
#             test_pbar = tqdm(self.test_loader, disable=not accelerator.is_local_main_process)
#             test_pbar.set_description(f'Test epoch {e}')
#
#             total_loss = 0
#             self.model.eval()
#             with torch.no_grad():
#                 for x, y in test_pbar:
#                     pred = self.model(x)
#                     loss = self.loss(pred, y)
#
#                     test_pbar.set_postfix(loss=loss.cpu().item())
#                     total_loss += loss.cpu().item()
#         return total_loss
