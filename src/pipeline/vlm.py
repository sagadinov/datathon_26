"""Optional VLM confirmation/description (Qwen2.5-VL via a local vLLM endpoint).

No-op unless VLM_URL is set and reachable: returns None so the caller keeps the rule template.
Never contacts anything but the configured local endpoint.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("pipeline.vlm")


def available() -> bool:
    return bool(os.environ.get("VLM_URL"))


def confirm(frames_b64: list[str], question: str, timeout: float = 20.0) -> dict | None:
    """Ask the local VLM; returns {"confirmed": bool, "description": str} or None when unavailable."""
    if not available():
        return None
    try:
        import requests

        url = os.environ["VLM_URL"].rstrip("/") + "/v1/chat/completions"
        content = [{"type": "text", "text": question}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in frames_b64]
        r = requests.post(url, json={"model": os.environ.get("VLM_MODEL", "qwen2.5-vl"),
                                     "messages": [{"role": "user", "content": content}], "max_tokens": 200},
                          timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return {"confirmed": not text.lower().startswith("нет"), "description": text}
    except Exception as e:  # degrade, never fail the job
        log.warning("vlm unavailable: %s", e)
        return None
