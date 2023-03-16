import optuna
from utils import parse_args, HyperSearcher


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