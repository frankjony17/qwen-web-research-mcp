"""MCP server exposing web scraping + local-LLM-powered analysis as tools."""
from __future__ import annotations

import logging
import os
import time

from mcp.server.mcpserver import Context, MCPServer

from . import ollama_client
from .analysis import analyze_text
from .scraper import FetchError, fetch_page_text
from .search import search_site

logger = logging.getLogger("web_research")

mcp = MCPServer("web-research")


@mcp.tool()
async def analyze_page(url: str, question: str, ctx: Context) -> str:
    """Fetch a web page and use a local LLM (via Ollama) to answer `question`
    about its full content. Long pages are chunked and analyzed piece by
    piece, so nothing is skipped -- can take 30s-3min. Reports progress while
    working.

    Args:
        url: The page to fetch and analyze.
        question: What to extract or answer about the page's content.
    """
    start = time.monotonic()
    logger.info("analyze_page start url=%s question=%r", url, question)
    try:
        page = await fetch_page_text(url, ctx=ctx)
    except FetchError as exc:
        logger.warning("analyze_page fetch failed url=%s error=%s", url, exc)
        return f"Error: {exc}"

    logger.info("analyze_page fetched url=%s chars=%d links=%d", url, len(page["text"]), len(page.get("links", [])))
    answer = await analyze_text(page["text"], question, ctx=ctx)
    title = page["title"] or url
    logger.info("analyze_page done url=%s elapsed=%.1fs", url, time.monotonic() - start)
    return f"# {title}\n{url}\n\n{answer}"


@mcp.tool()
async def search_site_and_analyze(
    site: str, phrase: str, question: str, ctx: Context, max_results: int | None = None
) -> str:
    """Search a site for pages/listings containing `phrase`, then run
    `analyze_page`-style extraction on each match to answer `question`.
    Works on any site: uses DuckDuckGo's `site:` search, no per-site code.
    Each match takes 30s-3min, so many matches take proportionally long.

    Args:
        site: Domain to search within, e.g. "example.com".
        phrase: Exact phrase the publication/listing must contain.
        question: What information to extract from each matching page.
        max_results: Max pages to analyze. Leave unset for no cap (as many
            as the search finds across all engines).
    """
    start = time.monotonic()
    logger.info("search_site_and_analyze start site=%s phrase=%r question=%r max_results=%s",
                site, phrase, question, max_results)
    matches = await search_site(site, phrase, max_results=max_results, ctx=ctx)
    logger.info("search_site_and_analyze found %d matches for site=%s", len(matches), site)
    if not matches:
        return f'No se encontraron resultados en {site} para la frase "{phrase}".'

    sections = []
    for idx, match in enumerate(matches, start=1):
        url = match["url"]
        if not url:
            continue
        logger.info("search_site_and_analyze processing %d/%d url=%s", idx, len(matches), url)
        await ctx.report_progress(idx - 1, len(matches), f"Procesando resultado {idx}/{len(matches)}: {url}")
        try:
            page = await fetch_page_text(url, ctx=ctx)
            answer = await analyze_text(page["text"], question, ctx=ctx)
            title = page["title"] or match["title"] or url
            sections.append(f"## {title}\n{url}\n\n{answer}")
        except FetchError as exc:
            logger.warning("search_site_and_analyze fetch failed url=%s error=%s", url, exc)
            sections.append(f"## {match.get('title') or url}\n{url}\n\nError: {exc}")

    logger.info("search_site_and_analyze done site=%s elapsed=%.1fs", site, time.monotonic() - start)
    return "\n\n---\n\n".join(sections)


@mcp.tool()
async def list_available_models() -> list[str]:
    """List the LLM models currently available on the local Ollama server."""
    logger.info("list_available_models called")
    return await ollama_client.list_models()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(
            transport=transport,
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
        )


if __name__ == "__main__":
    main()
