import torch
from utils import register_optimizer
import yaml

__all__ = ['load_optimizer']


def load_optimizer(model_params, cfg):
    with open(cfg.opt, 'r') as yaml_file:
        opt_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return register_optimizer[opt_cfg['name']](model_params, **parse_optimizer_kwargs(opt_cfg))


def parse_optimizer_kwargs(data_cfg: dict) -> dict:
    data_cfg.pop('name')
    params = {}
    for k, v in data_cfg.items():
        params[k] = float(v)
    return params


@register_optimizer
def Adam(model_params, lr, beta1, beta2, eps, weight_decay):
    return torch.optim.Adam(params=model_params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)


@register_optimizer
def SGD(model_params, lr, momentum, weight_decay):
    return torch.optim.SGD(params=model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
