"""Fetch a URL and extract its main readable content, regardless of site layout."""
from __future__ import annotations

import time

import httpx
import trafilatura

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 qwen-web-research-mcp/0.1"
)

# Only timeouts are retried (transient) -- a 404 or DNS failure won't fix itself.
MAX_TIMEOUT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0


class FetchError(RuntimeError):
    pass


def fetch_page_text(url: str, *, timeout: float = 20.0) -> dict:
    """Download a URL and extract clean article text + metadata.

    Uses trafilatura's extraction, which strips navigation/ads/boilerplate
    based on content heuristics rather than site-specific CSS selectors, so
    it keeps working if a site's markup changes.
    """
    response = None
    last_timeout_exc: httpx.TimeoutException | None = None
    for attempt in range(MAX_TIMEOUT_RETRIES + 1):
        try:
            response = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            break
        except httpx.TimeoutException as exc:
            last_timeout_exc = exc
            if attempt < MAX_TIMEOUT_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise FetchError(
                f"Failed to fetch {url}: timed out after {MAX_TIMEOUT_RETRIES + 1} attempts"
            ) from last_timeout_exc
        except httpx.HTTPError as exc:
            raise FetchError(f"Failed to fetch {url}: {exc}") from exc

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        output_format="markdown",
        with_metadata=True,
        url=url,
    )
    if not extracted:
        raise FetchError(f"Could not extract readable content from {url}")

    metadata = trafilatura.extract_metadata(response.text, default_url=url)

    return {
        "url": url,
        "title": metadata.title if metadata else None,
        "author": metadata.author if metadata else None,
        "date": metadata.date if metadata else None,
        "text": extracted,
    }
