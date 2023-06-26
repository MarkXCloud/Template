from abc import ABCMeta,abstractmethod

class _Evaluator(metaclass=ABCMeta):
    def __init__(self):
        self._buffer_pred = []
        self._buffer_label = []
    @abstractmethod
    def add_batch(self,pred,label):
        ...
    @abstractmethod
    def compute(self):
        ...