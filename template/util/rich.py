import os
from rich.console import Console,NewLine,JustifyMethod
from rich.style import Style
from rich.styled import Styled
from rich.scope import render_scope
from rich.segment import Segment
from rich.text import Text
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Any,
    Union,
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

install(show_locals=False)

from accelerate.state import PartialState

state = PartialState()

_singleton_dict = dict()


class MainConsole(Console):
    _single_dict = _singleton_dict

    def __init__(self, *args, **kwargs):
        self.__dict__ = self._single_dict
        if not self.initialized:
            super().__init__(*args, **kwargs)

    @state.on_main_process
    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)

    @state.on_main_process
    def log(
            self,
            *objects: Any,
            sep: str = " ",
            end: str = "\n",
            style: Optional[Union[str, Style]] = None,
            justify: Optional[JustifyMethod] = None,
            emoji: Optional[bool] = None,
            markup: Optional[bool] = None,
            highlight: Optional[bool] = None,
            log_locals: bool = False,
            _stack_offset: int = 1,
    ) -> None:
        """Log rich content to the terminal.

        Args:
            objects (positional args): Objects to log to the terminal.
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\\\\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            justify (str, optional): One of "left", "right", "center", or "full". Defaults to ``None``.
            overflow (str, optional): Overflow method: "crop", "fold", or "ellipsis". Defaults to None.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default. Defaults to None.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default. Defaults to None.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default. Defaults to None.
            log_locals (bool, optional): Boolean to enable logging of locals where ``log()``
                was called. Defaults to False.
            _stack_offset (int, optional): Offset of caller from end of call stack. Defaults to 1.
        """
        if not objects:
            objects = (NewLine(),)

        render_hooks = self._render_hooks[:]

        with self:
            renderables = self._collect_renderables(
                objects,
                sep,
                end,
                justify=justify,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
            )
            if style is not None:
                renderables = [Styled(renderable, style) for renderable in renderables]

            filename, line_no, locals = self._caller_frame_info(_stack_offset)
            link_path = None if filename.startswith("<") else os.path.abspath(filename)
            path = filename.rpartition(os.sep)[-1]
            if log_locals:
                locals_map = {
                    key: value
                    for key, value in locals.items()
                    if not key.startswith("__")
                }
                renderables.append(render_scope(locals_map, title="[i]locals"))

            renderables = [
                self._log_render(
                    self,
                    renderables,
                    log_time=self.get_datetime(),
                    path=path,
                    line_no=line_no,
                    link_path=link_path,
                )
            ]
            for hook in render_hooks:
                renderables = hook.process_renderables(renderables)
            new_segments: List[Segment] = []
            extend = new_segments.extend
            render = self.render
            render_options = self.options
            for renderable in renderables:
                extend(render(renderable, render_options))
            buffer_extend = self._buffer.extend
            for line in Segment.split_and_crop_lines(
                    new_segments, self.width, pad=False
            ):
                buffer_extend(line)

    @state.on_main_process
    def rule(self, *args, **kwargs):
        super().rule(*args, **kwargs)

    @property
    def initialized(self):
        return self._single_dict != {}


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
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """

    columns: List["ProgressColumn"] = (
        [SpinnerColumn(style="magenta", finished_text=":white_check_mark:")]
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
