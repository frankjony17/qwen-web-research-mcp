"""HTTP client for a local Ollama server running on the Windows host."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from functools import lru_cache

import httpx

logger = logging.getLogger("web_research.ollama")

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))


@lru_cache(maxsize=1)
def _detect_windows_host_ip() -> str:
    """IP WSL uses to reach the Windows host. Override with OLLAMA_HOST_IP."""
    override = os.environ.get("OLLAMA_HOST_IP")
    if override:
        return override

    result = subprocess.run(
        ["ip", "route", "show", "default"],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"default via (\S+)", result.stdout)
    if not match:
        raise RuntimeError(
            "Could not detect the Windows host IP from `ip route show default`. "
            "Set OLLAMA_HOST_IP explicitly."
        )
    return match.group(1)


def ollama_base_url() -> str:
    return f"http://{_detect_windows_host_ip()}:{OLLAMA_PORT}"


DEFAULT_CHAT_TIMEOUT = float(os.environ.get("OLLAMA_CHAT_TIMEOUT", "600"))


async def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_CHAT_TIMEOUT,
) -> str:
    """Send a single-turn chat request to Ollama and return the assistant's text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    logger.debug("chat request model=%s prompt_chars=%d", model, len(prompt))
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{ollama_base_url()}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        logger.debug("chat response model=%s elapsed=%.1fs answer_chars=%d", model, time.monotonic() - start, len(content))
        return content


async def list_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{ollama_base_url()}/api/tags")
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        logger.debug("list_models -> %s", models)
        return models
