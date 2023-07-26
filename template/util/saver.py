import shutil
from pathlib import Path
from dataclasses import dataclass
import datetime
from accelerate.tracking import on_main_process
from typing import Union
from template.util.rich import MainConsole

console = MainConsole(color_system='auto',log_time_format='[%Y.%m.%d %H:%M:%S]')


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
                 ):
        """
        :param higher_is_better: when we want to save the best model, we should point out what is 'best', higher_is_better means\
        if the metric we choose is higher, then we get a better model, so we save it!
        :param monitor: the metric that we want to observe for best model, e.g., accuracy
        :param save_dir: save direction, generate by generate_config_path()
        """

        self.save_dir = save_dir
        self.hib = higher_is_better
        self.monitor = monitor

        self.save_dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Current project save dir: {self.save_dir}")
        shutil.copy(src=config, dst=self.save_dir)
        self._metric = -1 if self.hib else 65535

    @on_main_process
    def save_best(self, metric):
        metric = metric[self.monitor]
        condition = metric > self._metric if self.hib else metric < self._metric
        if condition:
            self._metric = metric
            console.log(f"Save new [bold cyan]best model[/bold cyan] under {self.save_dir}")
            return True
        return False

    @classmethod
    def from_configuration(cls,  config: str,configuration: SaverConfiguration):
        save_dir = configuration.save_dir
        hib = configuration.higher_is_better
        monitor = configuration.monitor
        return cls(save_dir=save_dir, monitor=monitor, higher_is_better=hib,config=config)

    @property
    def best_metric(self):
        return self._metric

    def __repr__(self):
        return f"save directory:{self.save_dir}\n" \
               f"monitor: {self.monitor}\n" \
               f"high is better: {self.hib}\n" \
               f"current best metric: {self.best_metric}"


def generate_config_path(config: str, save_dir: str):
    """generate save direction by $CONFIG$/$current time$"""
    save_dir = Path(save_dir) / Path(config).stem / datetime.datetime.today().strftime(
        '%Y%m%d_%H_%M_%S')

    return save_dir
