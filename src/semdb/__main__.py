"""Entry point: ``python -m semdb`` (and the frozen SemanticDBSearch.exe).

``--selftest`` runs a headless end-to-end check (model, import, index, search) and exits
with status 0 on success; it is used to verify packaged builds.  ``--import-worker`` is the
memory-capped helper process that parses imported files (see core/isolation.py).
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--import-worker"]:
        from .core.isolation import worker_main

        return worker_main(argv[1:])
    if "--selftest" in argv:
        from .selftest import run_selftest

        return run_selftest(argv)
    from .ui.app import run_app

    return run_app(argv)


if __name__ == "__main__":
    raise SystemExit(main())
