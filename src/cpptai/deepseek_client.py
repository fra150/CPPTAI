"""Minimal DeepSeek API client using only the Python standard library.

This client targets the Chat Completions endpoint and defaults to model
"DeepSeek-V3.2-Exp" per user request. It reads the API key from the
environment variable `DEEPSEEK_API_KEY` and avoids external dependencies.
"""

from __future__ import annotations

import json
import os
import ssl
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from .env import load_env
from pathlib import Path
import hashlib


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"


def deepseek_chat(
    messages: List[Dict[str, str]],
    model: str = "DeepSeek-V3.2-Exp",
    stream: bool = False,
    base_url: str = DEEPSEEK_BASE_URL,
) -> Optional[Dict]:
    """Call DeepSeek Chat Completions API and return the parsed JSON response.

    Args:
        messages: Conversation messages in OpenAI-compatible format.
        model: Model name. Defaults to "DeepSeek-V3.2-Exp" per the request.
        stream: When True, asks the API to stream. This client does not handle
            streaming responses; the flag is forwarded as-is.
        base_url: API base URL. Default points to the official DeepSeek API.

    Returns:
        Parsed JSON dictionary on success, or None if a recoverable error occurs.
    """

    # Load .env once before reading variables.
    load_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # Fail gracefully if no key is present.
        return None

    url = f"{base_url}{CHAT_COMPLETIONS_PATH}"
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0,
        "seed": 0,
    }

    data = json.dumps(body, sort_keys=True).encode("utf-8")

    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    # Create a default SSL context; can be customized if needed.
    context = ssl.create_default_context()

    # Optional on-disk cache for determinism and rate safety
    use_cache = os.getenv("DEEPSEEK_CACHE", "0") == "1"
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256((url + data.decode("utf-8")).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"deepseek_{cache_key}.json"

    try:
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        with urlopen(req, context=context, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
            parsed = json.loads(payload)
            if use_cache:
                # Write cache atomically where possible
                try:
                    cache_path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return parsed
    except Exception:
        # Intentionally do not log secrets or full error details; return None.
        return None


def extract_text_answer(response: Dict) -> Optional[str]:
    """Extract the assistant text from a chat completions response.

    The function safely navigates the typical OpenAI-compatible structure and
    returns None if the expected fields are absent.
    """

    try:
        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        return message.get("content")
    except Exception:
        return None
