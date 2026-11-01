"""Background work: ``QRunnable`` tasks on a private ``QThreadPool`` (ARCHITECTURE.md §12.3).

A task function receives ``(progress, cancel)``: ``progress(message, fraction)`` is throttled
and forwarded to the UI thread, ``cancel()`` returns True once :meth:`Task.cancel` was called
(long operations poll it and raise :class:`Cancelled`).  Results, failures and cancellation
arrive on the UI thread through the task's signals holder.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..core.types import Cancelled, CancelFn, ProgressFn

log = logging.getLogger(__name__)

TaskFn = Callable[[ProgressFn, CancelFn], Any]


def describe_error(exc: BaseException) -> str:
    """A message suitable for the user; the traceback goes to the log."""
    from ..core.table import TableError

    if isinstance(exc, TableError):
        return str(exc)
    if isinstance(exc, MemoryError):
        return "There is not enough memory to finish this. Close other programs and try again."
    if isinstance(exc, PermissionError):
        return f"Access was denied: {exc.filename or exc}. Is the file open in another program?"
    if isinstance(exc, FileNotFoundError):
        return f"The file was not found: {exc.filename or exc}"
    if isinstance(exc, OSError):
        return f"The file could not be accessed: {exc.strerror or exc}"
    return f"Something went wrong ({type(exc).__name__}: {exc}). Details were written to the log."


class TaskSignals(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()  # always emitted last


class Task(QRunnable):
    """One unit of background work.  Python keeps it alive (see :class:`TaskRunner`)."""

    PROGRESS_INTERVAL = 0.05

    def __init__(self, fn: TaskFn, name: str = "task") -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.name = name
        self.signals = TaskSignals()
        self._fn = fn
        self._cancel = threading.Event()
        self._last_progress = 0.0
        self.error: BaseException | None = None

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def _progress(self, message: str, fraction: float) -> None:
        now = time.monotonic()
        if fraction < 1.0 and now - self._last_progress < self.PROGRESS_INTERVAL:
            return
        self._last_progress = now
        self.signals.progress.emit(message, float(fraction))

    def run(self) -> None:
        try:
            result = self._fn(self._progress, self.is_cancelled)
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # reported to the UI; must never escape into Qt
            self.error = exc
            log.exception("Background task %r failed", self.name)
            self.signals.failed.emit(describe_error(exc))
        else:
            self.signals.done.emit(result)
        finally:
            self.signals.finished.emit()


class _Relay(QObject):
    """Lives on the UI thread; forwards a task's signals to plain callbacks."""

    def __init__(
        self,
        on_done: Callable[[Any], None] | None,
        on_failed: Callable[[str], None] | None,
        on_progress: Callable[[str, float], None] | None,
        on_cancelled: Callable[[], None] | None,
        on_finished: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_done = on_done
        self._on_failed = on_failed
        self._on_progress = on_progress
        self._on_cancelled = on_cancelled
        self._on_finished = on_finished

    @Slot(object)
    def done(self, result: Any) -> None:
        if self._on_done:
            self._on_done(result)

    @Slot(str)
    def failed(self, message: str) -> None:
        if self._on_failed:
            self._on_failed(message)

    @Slot(str, float)
    def progress(self, message: str, fraction: float) -> None:
        if self._on_progress:
            self._on_progress(message, fraction)

    @Slot()
    def cancelled(self) -> None:
        if self._on_cancelled:
            self._on_cancelled()

    @Slot()
    def finished(self) -> None:
        self._on_finished()


class TaskRunner(QObject):
    """Starts tasks on a private thread pool and keeps them alive until they finish."""

    def __init__(self, parent: QObject | None = None, max_threads: int = 4) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_threads)
        self._active: dict[Task, _Relay] = {}

    def start(
        self,
        fn: TaskFn,
        *,
        name: str = "task",
        on_done: Callable[[Any], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> Task:
        task = Task(fn, name)

        def finished() -> None:
            self._active.pop(task, None)
            if on_finished:
                on_finished()

        relay = _Relay(on_done, on_failed, on_progress, on_cancelled, finished)
        task.signals.done.connect(relay.done)
        task.signals.failed.connect(relay.failed)
        task.signals.progress.connect(relay.progress)
        task.signals.cancelled.connect(relay.cancelled)
        task.signals.finished.connect(relay.finished)
        self._active[task] = relay
        self._pool.start(task)
        return task

    def active(self) -> list[Task]:
        return list(self._active)

    def cancel_all(self) -> None:
        for task in list(self._active):
            task.cancel()

    def wait(self, msecs: int = -1) -> bool:
        """Block until every started task returned (used at shutdown and in tests)."""
        return self._pool.waitForDone(msecs)
