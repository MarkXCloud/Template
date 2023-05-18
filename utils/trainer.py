from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import importlib
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchsummary import summary

accelerator = Accelerator(log_with=['wandb'], project_dir='./runs/')


class Trainer:
    """
    Trainer acts as a manager for the whole process of training, including loading every module, run log, init seed,
    train, test and save.
    """

    def __init__(self, config):
        # load every module
        cfg = config.cfg
        assert cfg.endswith('.py'), "cfg file should be a .py file"
        module = cfg.replace('/', '.')[:-3]
        module_loader = importlib.import_module(module)

        self.model = module_loader.model
        self.loss = module_loader.loss
        self.train_loader = module_loader.train_loader
        self.test_loader = module_loader.test_loader
        self.optimizer = module_loader.optimizer
        self.epoch = module_loader.epoch
        self.metric = module_loader.metric
        self.saver = module_loader.saver

        # basic info for the wandb log
        tracker_config = dict(
            epoch=self.epoch,
            model=self.model.__dict__.__getitem__('default_cfg')['architecture'],
            loss=self.loss.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
            learning_rate=self.optimizer.__dict__.__getitem__('param_groups')[0]['lr'],
            weight_decay=self.optimizer.__dict__.__getitem__('param_groups')[0]['weight_decay'],
            dataset=self.train_loader.__dict__.__getitem__('dataset'),
            batch_size=self.train_loader.__dict__.__getitem__('batch_size')
        )

        accelerator.init_trackers("example_project", config=tracker_config)

        device = accelerator.device
        self.model.to(device)
        self.model, self.optimizer, self.train_loader, self.test_loader \
            = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)

    def init_seeds(self, seed):
        # accelerate.set_seed() can seed all
        torch.cuda.empty_cache()
        set_seed(seed)
        torch.backends.cudnn.benchmark = True

    @accelerator.on_main_process
    def show_model_info(self):
        unwrapped_model = accelerator.unwrap_model(self.model)
        input_size = unwrapped_model.__dict__.__getitem__('default_cfg')['input_size']
        print(summary(model=unwrapped_model,
                      input_size=input_size))

    def train(self):
        for e in range(self.epoch):
            train_pbar = tqdm(self.train_loader, disable=not accelerator.is_local_main_process)
            train_pbar.set_description(f'Train epoch {e}')

            self.model.train()
            for x, y in train_pbar:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss(pred, y)

                accelerator.backward(loss)
                self.optimizer.step()

                train_pbar.set_postfix(loss=loss.cpu().item())

            test_pbar = tqdm(self.test_loader, disable=not accelerator.is_local_main_process)
            test_pbar.set_description(f'Test epoch {e}')

            self.model.eval()
            with torch.no_grad():
                for x, y in test_pbar:
                    pred = self.model(x)
                    loss = self.loss(pred, y)

                    test_pbar.set_postfix(loss=loss.cpu().item())

                    logit = torch.argmax(pred, dim=-1)
                    label = torch.argmax(y, dim=-1)
                    self.metric.add_batch(references=label, predictions=logit)

            acc = self.metric.compute()
            accelerator.print(acc)

            self.saver.save_latest_model(self.model)
            self.saver.save_best_model(self.model, acc)

            accelerator.log(acc, step=e)
        accelerator.end_training()


class HyperSearcher:
    def __init__(self, config):
        cfg = config.cfg
        assert cfg.endswith('.py'), "cfg file should be a .py file"
        module = cfg.replace('/', '.')[:-3]
        module_loader = importlib.import_module(module)

        self.model = module_loader.model
        self.loss = module_loader.loss
        self.train_set = module_loader.train_set
        self.test_set = module_loader.test_set
        self.epoch = module_loader.epoch
        assert self.epoch > 0
        self.optimizer, self.train_loader, self.test_loader = None, None, None

        device = accelerator.device
        self.model.to(device)

    def init_seeds(self, seed):
        # accelerate.set_seed() can seed all
        torch.cuda.empty_cache()
        set_seed(seed)
        torch.backends.cudnn.benchmark = True

    def hyp_search(self, trial):
        batch_size = trial.suggest_int('batch_size', 16, 64)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, step=0.0001)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        self.model, self.optimizer, self.train_loader, self.test_loader \
            = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)
        total_loss = 0
        for e in range(self.epoch):
            train_pbar = tqdm(self.train_loader, disable=not accelerator.is_local_main_process)
            train_pbar.set_description(f'Train epoch {e}')

            self.model.train()
            for x, y in train_pbar:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss(pred, y)

                accelerator.backward(loss)
                self.optimizer.step()

                train_pbar.set_postfix(loss=loss.cpu().item())

            test_pbar = tqdm(self.test_loader, disable=not accelerator.is_local_main_process)
            test_pbar.set_description(f'Test epoch {e}')

            total_loss = 0
            self.model.eval()
            with torch.no_grad():
                for x, y in test_pbar:
                    pred = self.model(x)
                    loss = self.loss(pred, y)

                    test_pbar.set_postfix(loss=loss.cpu().item())
                    total_loss += loss.cpu().item()
        return total_loss
