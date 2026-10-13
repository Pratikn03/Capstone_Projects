"""Enable ``python -m kga`` to invoke the command-line interface."""

from __future__ import annotations

from kga.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
