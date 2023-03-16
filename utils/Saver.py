from .trainer import accelerator
import os
class Saver:
    def __init__(self,save_step:int,higher_is_better:bool,monitor:str):
        self._save_dir = '.'
        self._save_step = save_step
        self._cnt = 0

        self.hib = higher_is_better
        self._metric = -1 if higher_is_better else 65535
        self.monitor = monitor

    def load_save_dir(self,save_dir:str):
        self._save_dir = save_dir

    def save_latest_model(self,model):
        if self._cnt==self._save_step:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self._save_dir ,"latest.pt"))
            self._cnt = 0
            accelerator.print("Save latest model!")
        else:
            self._cnt+=1

    def save_best_model(self,model,metric):
        metric = metric[self.monitor]
        condition = metric>self._metric if self.hib else metric < self._metric
        if condition:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f=os.path.join(self._save_dir,"/best.pt"))
            self._metric = metric
            accelerator.print("Save new best model!")
