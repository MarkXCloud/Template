import optuna
from template import parse_args
import importlib
import torch
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm

accelerator = Accelerator(log_with=['wandb'], project_dir='./runs/')

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
def main():
    cfg = parse_args()
    hyp_searcher = HyperSearcher(cfg)
    hyp_searcher.init_seeds(cfg.seed)

    study = optuna.create_study(study_name='test', direction='minimize')
    study.optimize(hyp_searcher.hyp_search, n_trials=5)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)


if __name__ == '__main__':
    main()