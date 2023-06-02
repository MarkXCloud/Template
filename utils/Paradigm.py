import torch


class BaseParadigm:
    """
    Paradigm shows a standard pipeline of certain deep learning task
    """
    def __init__(self):
        pass

    def train(self, model, x, y, loss_fn):
        raise NotImplementedError
    @torch.no_grad()
    def inference(self,model, x):
        raise NotImplementedError


class ImageClassificationParadigm(BaseParadigm):
    """
    In image classification paradigm, when we train a model, we get pred=model(x) and
    calculate loss(pred,y). And when we test a model, we get pred=model(x) and find the argmax
    of pred to get the class id.
    """
    def __init__(self):
        super().__init__()

    def train(self, model, x, y, loss_fn):
        return dict(loss=loss_fn(model(x), y))

    @torch.no_grad()
    def inference(self, model, x):
        return torch.argmax(model(x), dim=-1)
