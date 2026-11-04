"""JSON file boundary for the sequential teaching example.

Writes are ordinary file writes, not crash-safe atomic replacement. There is no
locking, concurrent update protection, or database transaction in this example.
"""

import json
from pathlib import Path

from workshop import validate_state


def load_workshop(path):
    """Read JSON and reject data that violates the workshop contract."""
    with Path(path).open(encoding="utf-8") as handle:
        state = json.load(handle)
    validate_state(state)
    return state


def save_workshop(path, state):
    """Validate before opening the destination, then save readable JSON."""
    validate_state(state)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
