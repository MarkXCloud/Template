import rich
from rich.console import Console
from rich.text import Text
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    ProgressColumn)

from rich.traceback import install

install(show_locals=True)

from accelerate.state import PartialState

state = PartialState()


class MainConsole(Console):
    @state.on_main_process
    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)

    @state.on_main_process
    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)

    @state.on_main_process
    def rule(self, *args, **kwargs):
        super().rule(*args, **kwargs)


class IterSpeedColumn(ProgressColumn):
    """Renders human readable iter speed."""

    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} iter/s", style="progress.data.speed")


def track(
        sequence,
        description: str = "Working...",
        total: Optional[float] = None,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style="bar.back",
        complete_style="bar.complete",
        finished_style="bar.finished",
        pulse_style="bar.pulse",
        update_period: float = 0.1,
        disable: bool = False
) -> Iterable:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        description (str, optional): Description of task show next to progress bar. Defaults to "Working".
        total: (float, optional): Total number of steps. Default is len(sequence).
        auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
        transient: (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
        style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
        pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
        update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
        disable (bool, optional): Disable display of progress.
        show_speed (bool, optional): Show speed if total isn't known. Defaults to True.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """

    columns: List["ProgressColumn"] = (
        [SpinnerColumn(style="magenta",finished_text=":white_check_mark:")]
    )
    columns.extend(
        [TextColumn("[progress.description]{task.description}")] if description else []
    )
    columns.extend(
        (
            BarColumn(
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
                bar_width=None,
            ),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            IterSpeedColumn(),
        )
    )
    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
        speed_estimate_period=5,
    )

    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=update_period
        )
