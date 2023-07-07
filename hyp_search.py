import optuna
from template import basic_args
import template.util as util
import torch
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm


class HyperSearcher:
    def __init__(self, args):
        self.args = args
        module_loader = util.load_module(args.config)
        self.model = module_loader.model
        self.loss_fn = module_loader.loss
        self.train_set = module_loader.train_set
        self.test_set = module_loader.test_set
        self.epoch = module_loader.epoch
        self.metric = module_loader.metric
        saver_config = module_loader.saver_config
        saver_config.save_dir = util.generate_config_path(config=args.config, save_dir=saver_config.save_dir)
        self.saver = util.Saver(configuration=saver_config, config=args.config)
        assert self.epoch > 0
        self.optimizer, self.train_loader, self.test_loader = None, None, None

    def hyp_search(self, trial):
        batch_size = trial.suggest_int('batch_per_gpu', 8, 64)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, step=0.0001)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.grad_step)
        accelerator.free_memory()
        torch.backends.cudnn.benchmark = True
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, collate_fn=self.train_set.collate_fn,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, collate_fn=self.test_set.collate_fn,
                                      shuffle=False, num_workers=self.args.num_workers)

        self.model, self.optimizer, self.train_loader, self.test_loader \
            = accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)
        for e in range(self.epoch):
            train_pbar = tqdm(self.train_loader, desc=f'Train epoch {e}', disable=not accelerator.is_local_main_process)

            self.model.train()
            self.loss_fn.train()
            for x, y in train_pbar:
                with accelerator.accumulate(self.model):
                    outputs = self.model(x)
                    loss_dict = self.loss_fn(outputs, y)
                    accelerator.backward(loss_dict['loss'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()


            test_pbar = tqdm(self.test_loader, desc=f'Test epoch {e}', disable=not accelerator.is_local_main_process)

            self.model.eval()
            self.loss_fn.eval()
            with torch.no_grad():
                for x, label in test_pbar:
                    outputs = self.model(x)
                    self.metric.add_batch(pred=outputs, label=label)

            metrics = self.metric.compute()
            self.saver.save_best(metric=metrics)
        return self.saver.best_metric

def main():
    args = basic_args()
    hyp_searcher = HyperSearcher(args)
    set_seed(args.seed)
    direction = 'maximize' if args.saver_config.higher_is_better else 'minimize'
    study = optuna.create_study(study_name='test', direction=direction)
    study.optimize(hyp_searcher.hyp_search, n_trials=5)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)


if __name__ == '__main__':
    main()
