from .trainer import accelerator
import os
import time


class Saver:
    """
    Saver acts as a scheduler to save the latest model and the best model.
    """

    def __init__(self, save_step: int, higher_is_better: bool, monitor: str):
        """
        :param save_step: when we want to save the latest model, it saves it per $save_step$ epochs.
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        """
        # create save dir
        save_dir = './runs/' + time.strftime('%Y%m%d_%H_%M_%S', time.gmtime(time.time()))
        os.mkdir(save_dir)
        self._save_dir = save_dir
        self._save_step = save_step
        # count for epochs, when the count meets save_step, it saves the latest model
        self._cnt = 0

        self.hib = higher_is_better
        self._metric = -1 if higher_is_better else 65535
        self.monitor = monitor

    def save_latest_model(self, model):
        if self._cnt == self._save_step:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self._save_dir, "latest.pt"))
            self._cnt = 0
            accelerator.print("Save latest model!")
        else:
            self._cnt += 1

    def save_best_model(self, model, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self._save_dir, "best.pt"))
            self._metric = metric
            accelerator.print("Save new best model!")
