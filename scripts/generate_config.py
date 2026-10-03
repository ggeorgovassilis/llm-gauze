#!/usr/bin/env python3
"""Regenerate ``.env.example`` and ``docs/configuration.md`` from ``config.py``.

Run from the repository root (or anywhere — paths are resolved relative to this
script). With ``--check`` it instead verifies the two files are up to date and
exits non-zero if they are not, for use in CI.
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


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = False
    for path, render in TARGETS.items():
        content = render()
        if path.exists() and path.read_text() == content:
            continue
        if check:
            print(f"{path.relative_to(REPO_ROOT)} is out of date — re-run scripts/generate_config.py")
            stale = True
        else:
            path.write_text(content)
            print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
