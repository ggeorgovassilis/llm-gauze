"""Rule-file helpers shared by the prompt-trim test modules.

``rules_document`` wraps a rule list in a version-1 rule document, and
``write_rules`` writes such a document into a temporary directory. Keeping
them in a plain support module (rather than ``conftest.py``) means test
modules do not have to import ``conftest`` directly.
"""

import json
from pathlib import Path
from typing import Any


def rules_document(rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a rule list in a version-1 prompt-trim rule document."""
    return {"version": 1, "rules": rules}


def write_rules(directory: Path, rules: list[dict[str, Any]]) -> Path:
    """Write a version-1 rule document into ``directory`` and return its path."""
    path = directory / "prompt_rules.json"
    path.write_text(json.dumps(rules_document(rules)), encoding="utf-8")
    return path
