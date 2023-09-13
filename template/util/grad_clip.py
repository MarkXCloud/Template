from abc import ABCMeta, abstractmethod
import torch


class GradClipper(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def clip(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class NormClipper(GradClipper):
    def __init__(self, parameters, max_norm, norm_type):
        super().__init__()
        self.parameters = parameters
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm, norm_type=self.norm_type)

    def __repr__(self):
        return f"""
        Gradient clip by {self.norm_type}-norm, max norm: {self.max_norm}"""
