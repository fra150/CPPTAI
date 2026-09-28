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
import urllib.error


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"

_MODEL_ALIASES = {
    "DeepSeek-V3.2-Exp": "deepseek-chat",
    "DeepSeek-V3.2": "deepseek-chat",
    "DeepSeek-V3": "deepseek-chat",
    "DeepSeek-R1": "deepseek-reasoner",
}


def _normalize_model_name(model: str) -> str:
    raw = (model or "").strip()
    if raw in _MODEL_ALIASES:
        return _MODEL_ALIASES[raw]
    return raw or "deepseek-chat"



def deepseek_chat(
    messages: List[Dict[str, str]],
    model: str = "DeepSeek-V3.2-Exp",
    stream: bool = False,
    base_url: str = DEEPSEEK_BASE_URL,
    temperature: float = 0,
    max_tokens: int = 2048,
    ) -> Optional[Dict]:
    """Call DeepSeek Chat Completions API and return the parsed JSON response.

    Args:
        messages: Conversation messages in OpenAI-compatible format.
        model: Model name. Defaults to "DeepSeek-V3.2-Exp" per the request.
        stream: When True, asks the API to stream. This client does not handle
            streaming responses; the flag is forwarded as-is.
        base_url: API base URL. Default points to the official DeepSeek API.
        temperature: Sampling temperature (0 = deterministic). Defaults to 0.
        max_tokens: Maximum tokens in the response. Defaults to 2048.

    Returns:
        Parsed JSON dictionary on success, or None if a recoverable error occurs.
    """

    # Load .env once before reading variables.
    load_env()
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        # Fail gracefully if no key is present.
        return None

    url = f"{base_url}{CHAT_COMPLETIONS_PATH}"
    normalized_model = _normalize_model_name(model)
    body = {
        "model": normalized_model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 0,
    }

    data = json.dumps(body, sort_keys=True).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    # Create a default SSL context; can be customized if needed.
    context = ssl.create_default_context()

    use_cache = os.getenv("DEEPSEEK_CACHE", "0") == "1"
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256((url + data.decode("utf-8")).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"deepseek_{cache_key}.json"

    import time as _time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if use_cache and cache_path.exists():
                return json.loads(cache_path.read_text(encoding="utf-8"))
            with urlopen(req, context=context, timeout=30) as resp:
                payload = resp.read().decode("utf-8")
                parsed = json.loads(payload)
                if use_cache:
                    try:
                        cache_path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                    except (IOError, OSError):
                        pass
                return parsed
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** attempt
                _time.sleep(wait)
                continue
            return None
        except (urllib.error.URLError, ssl.SSLError, json.JSONDecodeError, TimeoutError):
            if attempt < max_retries - 1:
                _time.sleep(1)
                continue
            return None
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
    except (KeyError, IndexError, TypeError) as e:
        return None
