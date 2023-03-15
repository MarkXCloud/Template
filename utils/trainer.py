from tqdm import tqdm
import torch
import os
import importlib
from accelerate import Accelerator
from accelerate.utils import set_seed

accelerator = Accelerator(project_dir="./runs")


class Trainer:
    def __init__(self, cfg):
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


        device = accelerator.device
        self.model.to(device)
        self.model, self.optimizer, self.train_loader, self.test_loader \
            = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)

    def init_seeds(self, seed):
        # accelerate.set_seed() can seed all
        torch.cuda.empty_cache()
        set_seed(seed)
        torch.backends.cudnn.benchmark = True

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
