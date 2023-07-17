import shutil
from pathlib import Path
from dataclasses import dataclass
import datetime
from accelerate.tracking import on_main_process
from typing import Union


@dataclass
class SaverConfiguration:
    higher_is_better: bool
    monitor: str
    save_dir: Union[Path | str] = Path('')


class Saver:
    """
    Saver can save the latest model and the best model.
    """
    main_process_only = True

    @on_main_process
    def __init__(self, higher_is_better: bool, monitor: str, save_dir: Union[Path | str], config: str,
                 configuration: SaverConfiguration = None):
        """
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        :param save_dir: save direction, generate by generate_config_path()
        """
        if configuration is not None:
            self.save_dir = configuration.save_dir
            self.hib = configuration.higher_is_better
            self.monitor = configuration.monitor
        else:
            self.save_dir = save_dir
            self.hib = higher_is_better
            self.monitor = monitor

        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Current project save dir: {self.save_dir}")
        shutil.copy(src=config, dst=self.save_dir)
        self._metric = -1 if self.hib else 65535

    @on_main_process
    def save_best(self, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            self._metric = metric
            print(f"Save new best model under {self.save_dir}")
            return True
        return False

    @property
    def best_metric(self):
        return self._metric


def generate_config_path(config: str, save_dir: str):
    """generate save direction by $CONFIG$/$current time$"""
    save_dir = Path(save_dir) / Path(config).stem / datetime.datetime.today().strftime(
        '%Y%m%d_%H_%M_%S')

    return save_dir
