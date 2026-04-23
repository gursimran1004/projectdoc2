from __future__ import annotations

import json
from pathlib import Path
from typing import List


HISTORY_PATH = Path(__file__).parent / ".upload_history.json"


def load_history() -> List[str]:
    if not HISTORY_PATH.exists():
        return []
    try:
        content = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(content, list):
            return [str(item) for item in content][-30:]
    except Exception:
        return []
    return []


def save_history(history: List[str]) -> None:
    trimmed = history[-30:]
    HISTORY_PATH.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
