import torch
from utils import parse_args, Trainer


def main():
    cfg = parse_args()
    trainer = Trainer(cfg.cfg)
    trainer.init_seeds(cfg.seed)
    trainer.train()


if __name__ == '__main__':
    main()
