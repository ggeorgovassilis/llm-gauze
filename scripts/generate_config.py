#!/usr/bin/env python3
"""Regenerate ``.env.example`` and ``docs/configuration.md`` from ``config.py``.

Run from the repository root (or anywhere — paths are resolved relative to this
script). The two files are derived from ``source/app/config.py``; drift from it
is also caught by the config-drift tests in ``tests/test_config.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from app.config_docs import configuration_md, env_example  # noqa: E402

TARGETS = {
    REPO_ROOT / ".env.example": env_example,
    REPO_ROOT / "docs" / "configuration.md": configuration_md,
}


def main() -> None:
    for path, render in TARGETS.items():
        path.write_text(render())
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
