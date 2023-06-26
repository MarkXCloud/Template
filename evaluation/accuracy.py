import torch
from accelerate.utils import gather
from ._BaseEvaluator import _Evaluator

__all__ = ['Accuracy']


class Accuracy(_Evaluator):
    def __init__(self):
        super().__init__()
        self.name = 'accuracy'

    def add_batch(self, pred, label):
        self._buffer_pred.append(pred)
        self._buffer_label.append(label)
    @torch.no_grad()
    def compute(self):
        pred = gather(torch.cat(self._buffer_pred, dim=0))
        label = gather(torch.cat(self._buffer_label, dim=0))
        num_correct = pred.eq_(label).float().sum().cpu().item()
        num = label.shape[0]
        self.buffer_pred = []
        self.buffer_label = []
        return dict(accuracy=num_correct / num)
