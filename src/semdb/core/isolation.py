"""Opening untrusted spreadsheet files in a separate, memory-capped worker process.

A crafted workbook can make a parser allocate gigabytes (a sheet that declares billions of
cells) or crash it outright.  The app therefore never parses an imported file in its own
process: :func:`run` starts a short-lived copy of itself (``--import-worker``), the worker
caps its own memory (a Windows Job Object; ``RLIMIT_AS`` elsewhere) before touching the file,
parses it with :mod:`semdb.core.table`, and writes the result as JSON (never pickle) to a
temporary folder.  If the worker dies or exceeds the cap, the app shows a friendly error and
stays responsive.  Progress travels through a small file the app polls.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .types import CancelFn, Cancelled, ProgressFn

log = logging.getLogger(__name__)

IN_PROCESS_ENV = "SEMDB_IMPORT_IN_PROCESS"  # "1": parse in this process (set inside the worker; tests)
MEMORY_ENV = "SEMDB_IMPORT_MEMORY_MB"  # overrides the worker's memory cap
TEST_OPS_ENV = "SEMDB_WORKER_TEST_OPS"  # "1": enables the test-only operations below
WORKER_FLAG = "--import-worker"
TIMEOUT_SECONDS = 15 * 60
_POLL_SECONDS = 0.1
_GIB = 1 << 30


class WorkerError(Exception):
    """No result from the worker. ``kind``: "table" (``message`` is the reader's own
    user-facing explanation), "memory" (hit the cap), "crashed" or "timeout"."""

    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind
        self.message = message


def in_process() -> bool:
    return os.environ.get(IN_PROCESS_ENV) == "1"


def memory_limit_bytes() -> int:
    """The worker's memory cap: 40 % of physical memory, between 1 and 3 GiB."""
    override = os.environ.get(MEMORY_ENV)
    if override and override.isdigit():
        return int(override) << 20
    total = _physical_memory()
    return int(min(3 * _GIB, max(_GIB, 0.4 * total))) if total else 2 * _GIB


def run(op: str, payload: dict[str, Any], progress: ProgressFn | None = None,
        cancel: CancelFn | None = None, timeout: float = TIMEOUT_SECONDS) -> Any:
    """Run ``op`` in a worker process and return its JSON result.

    Raises :class:`Cancelled` when ``cancel`` asks to stop and :class:`WorkerError` for every
    failure (including the reader's own user-facing errors, ``kind == "table"``).
    """
    with tempfile.TemporaryDirectory(prefix="semdb-import-", ignore_cleanup_errors=True) as folder:
        work = Path(folder)
        request, result, progress_file, errors = (work / name for name in
                                                   ("request.json", "result.json", "progress", "stderr.txt"))
        request.write_text(json.dumps({"op": op, "payload": payload}), encoding="utf-8")
        command, env = _worker_command(request, result, progress_file)
        with errors.open("wb") as stderr:
            process = subprocess.Popen(
                command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            code = _wait(process, progress_file, progress, cancel, timeout)
        answer = _read_json(result)
        if answer is not None and answer.get("ok"):
            return answer["value"]
        if answer is not None and answer.get("kind") in ("table", "memory"):
            raise WorkerError(answer["kind"], answer.get("error", ""))
        tail = errors.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
        log.warning("Import worker for %s exited with code %s without a result. %s", op, code, tail)
        # A cap hit inside native code aborts the process instead of raising MemoryError.
        raise WorkerError("memory" if _looks_like_memory_abort(code, tail) else "crashed")


def worker_main(argv: list[str]) -> int:
    """Entry point of ``--import-worker <request.json> <result.json> <progress file>``."""
    os.environ[IN_PROCESS_ENV] = "1"
    request_path, result_path, progress_path = (Path(arg) for arg in argv[:3])
    try:
        _limit_own_memory(memory_limit_bytes())
    except OSError as exc:  # isolation still protects the app; only the cap is missing
        print(f"memory cap not applied: {exc}", file=sys.stderr)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    last_write = 0.0

    def report(stage: str, fraction: float) -> None:
        nonlocal last_write
        now = time.monotonic()
        if now - last_write >= 0.1 or fraction >= 1.0:
            last_write = now
            try:
                progress_path.write_text(f"{fraction:.4f}", encoding="ascii")
            except OSError:
                pass

    try:
        value = _execute(request["op"], request["payload"], report)
        answer: dict[str, Any] = {"ok": True, "value": value}
    except MemoryError:
        answer = {"ok": False, "kind": "memory", "error": "memory"}
    except Exception as exc:  # noqa: BLE001 - every failure must reach the app as a message
        from .table import TableError

        if isinstance(exc, TableError):
            answer = {"ok": False, "kind": "table", "error": str(exc)}
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            answer = {"ok": False, "kind": "crashed", "error": "crashed"}
    partial = result_path.with_suffix(".partial")
    partial.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, result_path)
    return 0 if answer["ok"] else 1


# --------------------------------------------------------------------------------------


def _execute(op: str, payload: dict[str, Any], report: ProgressFn) -> Any:
    from . import table

    if op == "list_sheets":
        return table.list_sheets(payload["path"])
    if op == "read_raw":
        return table.read_raw(payload["path"], payload.get("sheet"), payload.get("max_rows"))
    if op == "read_table":
        result = table.read_table(payload["path"], payload.get("sheet"), payload.get("header_row"), progress=report)
        return {"source_name": result.source_name, "sheet_name": result.sheet_name, "columns": result.columns,
                "rows": result.rows, "source_rows": result.source_rows}
    if os.environ.get(TEST_OPS_ENV) == "1":
        if op == "allocate":  # proves the memory cap: a huge allocation must fail
            block = bytearray(int(payload["bytes"]))
            return len(block)
        if op == "sleep":
            time.sleep(float(payload["seconds"]))
            return None
        if op == "exit":
            os._exit(int(payload["code"]))
    raise ValueError(f"unknown worker operation {op!r}")


def _worker_command(request: Path, result: Path, progress: Path) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ)
    env.pop(IN_PROCESS_ENV, None)
    files = [str(request), str(result), str(progress)]
    if getattr(sys, "frozen", False):
        return [sys.executable, WORKER_FLAG, *files], env
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
    return [sys.executable, "-s", "-m", "semdb", WORKER_FLAG, *files], env


def _wait(process: subprocess.Popen, progress_file: Path, progress: ProgressFn | None,
          cancel: CancelFn | None, timeout: float) -> int:
    from .table import READING_STAGE

    deadline = time.monotonic() + timeout
    while True:
        try:
            return process.wait(timeout=_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        if cancel is not None and cancel():
            _stop(process)
            raise Cancelled()
        if time.monotonic() > deadline:
            _stop(process)
            raise WorkerError("timeout")
        if progress is not None:
            try:
                progress(READING_STAGE, min(max(float(progress_file.read_text(encoding="ascii")), 0.0), 1.0))
            except (OSError, ValueError):
                pass


def _looks_like_memory_abort(code: int, stderr: str) -> bool:
    """Rust aborts on allocation failure ("memory allocation of N bytes failed")."""
    return "memory allocation" in stderr or "MemoryError" in stderr or (code & 0xFFFFFFFF) == 0xC0000017


def _stop(process: subprocess.Popen) -> None:
    process.kill()
    process.wait()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _physical_memory() -> int:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32), ("total_phys", ctypes.c_uint64),
                        ("avail_phys", ctypes.c_uint64), ("total_page", ctypes.c_uint64),
                        ("avail_page", ctypes.c_uint64), ("total_virtual", ctypes.c_uint64),
                        ("avail_virtual", ctypes.c_uint64), ("avail_extended", ctypes.c_uint64)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        return status.total_phys if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else 0
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0


def _limit_own_memory(limit: int) -> None:
    """Cap this process's committed memory at ``limit`` bytes (allocations beyond it fail)."""
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("process_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("reads", "writes", "others", "read_bytes",
                                                         "write_bytes", "other_bytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    limits.process_memory = limit
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):  # extended info
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # The job lives as long as this process; its handle is deliberately never closed.
