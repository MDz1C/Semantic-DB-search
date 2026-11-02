"""PyInstaller entry point for Fundscope.exe.

PyInstaller runs its entry script as a top-level ``__main__`` module, where the relative
imports of ``src/semdb/__main__.py`` would fail; importing the package from here keeps them
working.  Behaves exactly like ``python -m semdb``.
"""

from semdb.__main__ import main

raise SystemExit(main())
