from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional
from prettytable import PrettyTable
import json
from pathlib import Path


class SysTracker(GeneralTracker):
    name = "sys"
    requires_logging_directory = False
    main_process_only = True
    _buffer = []

    @on_main_process
    def __init__(self, logdir: Path):
        super().__init__()
        self.table = PrettyTable()
        self.table.float_format = ".6"
        self.logdir = logdir

    @property
    def tracker(self):
        return self.table

    @on_main_process
    def store_init_configuration(self, values: dict):
        pass

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        if not self.table.field_names:
            self.table.field_names = values.keys()
        self.table.add_row(values.values())
        print(self.table)
        self.table.clear_rows()

        values.update({'step': step})
        SysTracker._buffer.append(values)

    @on_main_process
    def finish(self):
        with open(self.logdir / 'result.json', "w") as f:
            json.dump(SysTracker._buffer, f)
