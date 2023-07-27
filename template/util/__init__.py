from .misc import *

def load_module(script_path):
    """
    This func is equal to 'import XXX' where XXX is the path to the .py file
    :param script_path: path to the .py config file
    :return: a module loader
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("module_script", script_path)
    module_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_script)

    return module_script