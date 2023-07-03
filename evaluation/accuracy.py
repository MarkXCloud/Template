import torch
from accelerate.utils import reduce
from ._BaseEvaluator import _Evaluator
from collections import OrderedDict


class Accuracy(_Evaluator):
    def __init__(self,topk=(1,)):
        super().__init__()
        self._buffer_pred = []
        self._buffer_label = []
        self.topk = topk

    def add_batch(self, pred, label):
        self._buffer_pred.append(pred)
        self._buffer_label.append(label)
    @torch.no_grad()
    def compute(self):
        pred = torch.cat(self._buffer_pred, dim=0)
        label = torch.cat(self._buffer_label, dim=0)

        maxk = max(self.topk)
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        num = reduce(torch.tensor(label.shape[0],device=label.device),reduction='sum')
        res = OrderedDict()
        for k in self.topk:
            correct_k = reduce(correct[:k].float().sum(),reduction='sum')
            res.update({f"accuracy@top{k}": correct_k.div_(num).item()})

        self._buffer_pred = []
        self._buffer_label = []
        return res