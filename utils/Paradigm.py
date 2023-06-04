import torch
from typing import Callable


class Paradigm:
    """
    Paradigm shows a standard pipeline of certain deep learning task

    train perform the whole pipeline of training from pred=model(x) to loss=loss_fn(pred,y)
    :return: the loss dict which contains the value of 'loss' and other losses if existed.

    test perform the whole pipeline of inference from pred=model(x) to results=convert_to_label(pred).
    :return: the prediction of model in the form of corresponding label, e.g., one-hot vector -> class index
    Note: most of the time, we should use @torch.no_grad() decorator on inference function
    """

    def __init__(self, train: Callable, test: Callable):
        self.train = train
        self.test = test


def image_classification_train(model, x, y, loss_fn):
    return dict(loss=loss_fn(model(x), y))


@torch.no_grad()
def image_classification_inference(model, x):
    return torch.argmax(model(x), dim=-1)


ImageClassificationParadigm = Paradigm(image_classification_train, image_classification_inference)
