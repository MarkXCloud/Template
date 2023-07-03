import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict

class ClsCrossEntropy(nn.CrossEntropyLoss):
    name = 'CrossEntropy'


    def forward(self, input: Tensor, target: Tensor) -> dict:
        return OrderedDict(
            loss=F.cross_entropy(input, target, weight=self.weight,
                                 ignore_index=self.ignore_index, reduction=self.reduction,
                                 label_smoothing=self.label_smoothing)
        )


class ClsMSE(nn.MSELoss):
    name = 'MSE'

    def forward(self, input: Tensor, target: Tensor) -> dict:
        return OrderedDict(
            loss=F.mse_loss(input, target, reduction=self.reduction)
        )

