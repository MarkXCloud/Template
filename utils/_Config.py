import importlib
from prettytable import PrettyTable


class Config(dict):
    """
    Config is for storing all kwargs from the cfg.py.
    It contains:
    model_params for model configuration,
    loss_params for loss configuration,
    train_set_params for training set configuration,
    test_set_params for testing set configuration,
    optimizer_params for optimizer configuration,
    trainer_params for trainer configuration
    """
    def __init__(self, module: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert module.endswith('.py'), "cfg file should be a .py file"
        module = module.replace('/', '.')[:-3]
        module_loader = importlib.import_module(module)
        # self.module = module_loader
        # self.model_params = module_loader.model
        # self.loss_params = module_loader.loss
        # self.train_set_params = module_loader.train_set
        # self.test_set_params = module_loader.test_set
        # self.optimizer_params = module_loader.optimizer
        # self.trainer_params = module_loader.trainer

    def __repr__(self):
        pass
        # tb = PrettyTable()
        # tb.field_names = ["Attribute","Value"]
        # tb.add_row(["Model",self.model_params['model_name']])
        # tb.add_row(["Loss",self.loss_params['name']])
        # tb.add_row(["Dataset",self.train_set_params['name']])
        # tb.add_row(["Optimizer",self.optimizer_params['name']])
        # tb.add_row(["Trainer",self.trainer_params['paradigm']])
        # tb.add_row(["Epoch",self.trainer_params['epoch']])
        # return tb.get_string()
