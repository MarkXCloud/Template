import json
from pathlib import Path
from typing import Optional
from accelerate.tracking import GeneralTracker, on_main_process
from rich.console import Console
from rich.table import Table,Column

console = Console(color_system='auto')


class SysTracker(GeneralTracker):
    name = "sys"
    requires_logging_directory = False
    main_process_only = True
    _buffer = []

    @on_main_process
    def __init__(self, logdir: Path):
        super().__init__()
        self.logdir = logdir

    @property
    def tracker(self):
        return SysTracker._buffer

    @on_main_process
    def store_init_configuration(self, values: dict):
        ...

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        table = Table(*[Column(header=k,justify='center',header_style="bold magenta") for k in values.keys()],title="Metrics")
        table.add_row(*[f"{v:.6f}" for v in values.values()],style='cyan')
        console.print(table,justify="center")
        values.update({'step': step})
        SysTracker._buffer.append(values)

    @on_main_process
    def finish(self):
        with open(self.logdir / 'result.json', "w") as f:
            json.dump(SysTracker._buffer, f)
