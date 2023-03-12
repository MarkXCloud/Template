import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torchvision
from torchvision import transforms
import yaml
from utils import register_dataset

__all__ = ['load_dataset']


def load_dataset(cfg):
    with open(cfg.data, 'r') as yaml_file:
        data_cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return *register_dataset[data_cfg['name']](**parse_dataset_kwargs(cfg, data_cfg)), data_cfg['cls']


def parse_dataset_kwargs(cfg, data_cfg: dict) -> dict:
    params = {}

    if 'root' in data_cfg.keys():
        params['root'] = data_cfg['root']
    if 'transforms' in data_cfg.keys():
        params['transforms'] = eval(data_cfg['transforms'])
    if 'target_transforms' in data_cfg.keys():
        params['target_transforms'] = eval(data_cfg['target_transforms'])

    if hasattr(cfg, 'dist'):
        params['is_distributed'] = cfg.dist
    if hasattr(cfg, 'batch_size'):
        params['batch_size'] = cfg.batch_size
    if hasattr(cfg, 'seed'):
        params['seed'] = cfg.seed
    if hasattr(cfg, 'num_workers'):
        params['num_workers'] = cfg.num_workers

    return params


def dataset2loader(dataset, batch_size, seed,num_workers, is_distributed=False):
    sampler = DistributedSampler(dataset, seed=seed) if is_distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,num_workers=num_workers)
    return loader


@register_dataset
def cifar10(root, batch_size, transforms, target_transforms, seed,num_workers, is_distributed=False):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms,
                                             target_transform=target_transforms)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms,
                                            target_transform=target_transforms)
    train_loader = dataset2loader(train_set, batch_size=batch_size, is_distributed=is_distributed, seed=seed,num_workers=num_workers)
    test_loader = dataset2loader(test_set, batch_size=1, is_distributed=is_distributed, seed=seed,num_workers=num_workers)
    # need:
    # for epoch in range(start_epoch, n_epochs):
    # 	if is_distributed:
    #         sampler.set_epoch(epoch)
    #         train(loader)
    return train_loader, test_loader
