"""Site-scoped phrase search via DuckDuckGo, so scraping stays site-agnostic."""
from __future__ import annotations

import asyncio
import logging

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from mcp.server.mcpserver import Context

logger = logging.getLogger("qwen_web_research.search")

# backend="auto" spreads the search across multiple engines to avoid blocks.
MAX_SEARCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0


async def search_site(
    site: str, phrase: str, *, max_results: int | None = None, ctx: Context | None = None
) -> list[dict]:
    """Find pages on `site` that mention `phrase`, using DuckDuckGo's `site:` operator.

    Works for any domain without site-specific scraping code. max_results=None
    (the default) means no cap -- ddgs returns everything it finds.
    """
    domain = site.replace("https://", "").replace("http://", "").strip("/")
    query = f'site:{domain} "{phrase}"'
    logger.info("search start query=%r max_results=%s", query, max_results)

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
                logger.warning("search rate-limited/timeout query=%r attempt=%d, retrying in %.0fs",
                                query, attempt + 1, wait)
                if ctx:
                    await ctx.report_progress(
                        attempt + 1,
                        MAX_SEARCH_RETRIES + 1,
                        f"Búsqueda bloqueada/timeout, reintentando en {wait:.0f}s...",
                    )
                await asyncio.sleep(wait)
                continue
            logger.error("search failed query=%r after %d attempts: %s", query, MAX_SEARCH_RETRIES + 1, exc)
            raise
    else:
        raise last_exc  # unreachable, satisfies type checkers

    logger.info("search done query=%r results=%d", query, len(results))
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]
