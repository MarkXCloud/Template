import torch
import torch.nn as nn
from typing import Callable

__all__ = ['LossCls']


class LossCls(nn.Module):
    def __init__(self, loss_fn: Callable):
        super().__init__()
        self.loss_fn = loss_fn
        self.name = loss_fn.__class__.__name__

    def __call__(self, *args, **kwargs) -> dict:
        return dict(loss=self.loss_fn(*args, **kwargs))
    @staticmethod
    def post_process(pred:torch.tensor):
        return torch.argmax(pred, dim=-1)
