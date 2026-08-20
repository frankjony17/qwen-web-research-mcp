"""Site-scoped phrase search via DuckDuckGo, so scraping stays site-agnostic."""
from __future__ import annotations

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

# backend="auto" already spreads the search across multiple engines (not just
# DuckDuckGo), which helps avoid a single provider's block. This adds retry
# with backoff on top, for when even that gets rate-limited.
MAX_SEARCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0


def search_site(site: str, phrase: str, *, max_results: int = 5) -> list[dict]:
    """Find pages on `site` that mention `phrase`, using DuckDuckGo's `site:` operator.

    Works for any domain without site-specific scraping code, since it relies on
    the site already being indexed by the search engine rather than a custom
    search integration per site.
    """
    domain = site.replace("https://", "").replace("http://", "").strip("/")
    query = f'site:{domain} "{phrase}"'

    last_exc: DDGSException | None = None
    for attempt in range(MAX_SEARCH_RETRIES + 1):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend="auto"))
            break
        except (RatelimitException, TimeoutException) as exc:
            last_exc = exc
            if attempt < MAX_SEARCH_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    else:
        raise last_exc  # unreachable, satisfies type checkers

    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]
