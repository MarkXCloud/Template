import torch
from ._BaseEvaluator import _Evaluator
from collections import OrderedDict


class Accuracy(_Evaluator):
    def __init__(self, topk=(1,)):
        super().__init__()
        self._buffer_pred = []
        self._buffer_label = []
        self.topk = topk

    def add_batch(self, pred, label) -> None:
        self._buffer_pred.append(pred.cpu())
        self._buffer_label.append(label.cpu())

    @torch.no_grad()
    def compute(self) -> OrderedDict:
        pred = torch.cat(self._buffer_pred, dim=0)
        label = torch.cat(self._buffer_label, dim=0)
        assert len(pred) == len(
            label), f"nums of data mismatch, pred has length {len(pred)} while label has length {len(label)}"

        maxk = max(self.topk)
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        num = label.shape[0]
        res = OrderedDict()
        for k in self.topk:
            correct_k = correct[:k].float().sum()
            res.update({f"accuracy@top{k}": correct_k.div_(num).item()})

        self._buffer_pred = []
        self._buffer_label = []
        return res
