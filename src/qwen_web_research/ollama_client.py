"""HTTP client for a local Ollama server (running on Windows, reached from WSL via the NAT gateway)."""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache

import httpx

DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "qwen3:14b")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))


@lru_cache(maxsize=1)
def _detect_windows_host_ip() -> str:
    """Return the IP WSL uses to reach the Windows host in NAT mode.

    Explicit OLLAMA_HOST_IP overrides detection. Otherwise this reads the
    default route, which points at the Windows host's vEthernet (WSL) adapter.
    """
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


def chat(prompt: str, *, system: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> str:
    """Send a single-turn chat request to Ollama and return the assistant's text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{ollama_base_url()}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


def list_models() -> list[str]:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{ollama_base_url()}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
