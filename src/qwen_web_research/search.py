"""Site-scoped phrase search via DuckDuckGo, so scraping stays site-agnostic."""
from __future__ import annotations

import asyncio

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from mcp.server.mcpserver import Context

# backend="auto" already spreads the search across multiple engines (not just
# DuckDuckGo), which helps avoid a single provider's block. This adds retry
# with backoff on top, for when even that gets rate-limited.
MAX_SEARCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0


async def search_site(
    site: str, phrase: str, *, max_results: int = 5, ctx: Context | None = None
) -> list[dict]:
    """Find pages on `site` that mention `phrase`, using DuckDuckGo's `site:` operator.

    Works for any domain without site-specific scraping code, since it relies on
    the site already being indexed by the search engine rather than a custom
    search integration per site.
    """
    domain = site.replace("https://", "").replace("http://", "").strip("/")
    query = f'site:{domain} "{phrase}"'

    last_exc: DDGSException | None = None
    results: list[dict] = []
    for attempt in range(MAX_SEARCH_RETRIES + 1):
        try:
            def _search() -> list[dict]:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results, backend="auto"))

            results = await asyncio.to_thread(_search)
            break
        except (RatelimitException, TimeoutException) as exc:
            last_exc = exc
            if attempt < MAX_SEARCH_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                if ctx:
                    await ctx.report_progress(
                        attempt + 1,
                        MAX_SEARCH_RETRIES + 1,
                        f"Búsqueda bloqueada/timeout, reintentando en {wait:.0f}s...",
                    )
                await asyncio.sleep(wait)
                continue
            raise
    else:
        raise last_exc  # unreachable, satisfies type checkers

    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]
