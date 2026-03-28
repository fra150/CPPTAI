from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _normalize_payload(payload: Any) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(payload)


def _key_to_path(cache_dir: Path, namespace: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{namespace}_{h}.json"


def cache_get(cache_dir: str, namespace: str, key_payload: Any) -> Optional[Dict[str, Any]]:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    key = _normalize_payload(key_payload)
    path = _key_to_path(cdir, namespace, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_set(cache_dir: str, namespace: str, key_payload: Any, value: Dict[str, Any]) -> None:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    key = _normalize_payload(key_payload)
    path = _key_to_path(cdir, namespace, key)
    try:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return
