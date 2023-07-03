from abc import ABCMeta,abstractmethod

class _Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def add_batch(self,pred,label):
        ...
    @abstractmethod
    def compute(self):
        ...