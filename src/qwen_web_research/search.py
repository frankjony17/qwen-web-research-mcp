"""Site-scoped phrase search via DuckDuckGo, so scraping stays site-agnostic."""
from __future__ import annotations

from ddgs import DDGS


def search_site(site: str, phrase: str, *, max_results: int = 5) -> list[dict]:
    """Find pages on `site` that mention `phrase`, using DuckDuckGo's `site:` operator.

    Works for any domain without site-specific scraping code, since it relies on
    the site already being indexed by the search engine rather than a custom
    search integration per site.
    """
    domain = site.replace("https://", "").replace("http://", "").strip("/")
    query = f'site:{domain} "{phrase}"'

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]
