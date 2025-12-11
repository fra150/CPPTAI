"""Simple .env loader without external dependencies.

Reads a `.env` file from the project root and sets process environment
variables. Lines beginning with `#` are treated as comments; blank lines are
ignored. Only KEY=VALUE pairs are supported.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_env(filename: str = ".env") -> Dict[str, str]:
    """Load environment variables from a .env-style file.

    Returns a mapping of keys loaded. Existing environment variables are not
    overwritten.
    """
    loaded: Dict[str, str] = {}
    root = Path.cwd()
    path = root / filename
    if not path.exists():
        return loaded
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and (key not in os.environ):
                os.environ[key] = value
                loaded[key] = value
    except Exception:
        # Fail silently to avoid breaking runtime when .env is malformed.
        return loaded
    return loaded
