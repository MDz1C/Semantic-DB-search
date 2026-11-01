"""Application entry point: offline guard, logging, crash dialog, theme and the main window."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from .. import APP_ID, APP_NAME, __version__, paths

log = logging.getLogger("semdb")

_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def setup_logging() -> None:
    """Log to ``logs/app.log`` in the data folder (1 MB x 3 files)."""
    target = paths.logs_dir() / "app.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(target):
            return
    handler = RotatingFileHandler(target, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


def _qt_messages(mode: QtMsgType, _context, message: str) -> None:
    logging.getLogger("qt").log(_QT_LEVELS.get(mode, logging.WARNING), message)


def install_excepthook() -> None:
    """Log unexpected errors and show a friendly dialog instead of crashing."""
    showing = False

    def hook(exc_type, exc, tb) -> None:
        nonlocal showing
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        if showing or QApplication.instance() is None:
            return
        showing = True
        try:
            from .widgets import show_error

            summary = "".join(traceback.format_exception_only(exc_type, exc)).strip()
            show_error(
                QApplication.activeWindow(),
                "Something went wrong",
                "The problem was written to the log file. You can keep working; if it happens again, "
                "restart the app.",
                summary,
            )
        except Exception:
            log.exception("Could not show the error dialog")
        finally:
            showing = False

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        log.critical("Unhandled exception in thread %s", args.thread,
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = hook
    threading.excepthook = thread_hook


def _set_windows_app_id() -> None:
    """Group the taskbar button under the app's own icon instead of Python's."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_ID}.App")
        except (AttributeError, OSError):
            log.debug("AppUserModelID not set", exc_info=True)


def run_app(argv: list[str]) -> int:
    """Start the desktop app and return its exit code."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    setup_logging()
    log.info("Starting %s %s", APP_NAME, __version__)
    _set_windows_app_id()
    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else APP_NAME, *argv])
    app.setOrganizationName(APP_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    qInstallMessageHandler(_qt_messages)
    install_excepthook()

    from . import icons, theme
    from .main_window import MainWindow
    from .widgets import show_error

    theme.manager().install(app)
    app.setWindowIcon(icons.app_icon())
    try:
        from ..storage import Store

        store = Store(paths.data_dir())
    except Exception as exc:
        log.exception("Could not open the data folder")
        show_error(None, f"{APP_NAME} could not start", f"The data folder could not be opened: {exc}")
        return 1
    theme.manager().set_mode(store.get_setting("theme", "system") or "system")
    window = MainWindow(store)
    window.show()
    try:
        return app.exec()
    finally:
        store.close()
        log.info("Stopped")
