"""MCP server exposing web scraping + Qwen-powered analysis as tools."""
from __future__ import annotations

import logging
import os
import time

from mcp.server.mcpserver import Context, MCPServer

from . import ollama_client
from .analysis import analyze_text
from .scraper import FetchError, fetch_page_text
from .search import search_site

logger = logging.getLogger("qwen_web_research")

mcp = MCPServer("qwen-web-research")


@mcp.tool()
async def analyze_page(url: str, question: str, ctx: Context) -> str:
    """Fetch a web page, and use a local Qwen model to extract/answer `question`
    about its full content. Long pages are automatically split into chunks and
    analyzed piece by piece (map-reduce), so no content is skipped.

    Reports progress while working (page fetch retries, per-chunk analysis),
    so a client that respects MCP progress notifications won't time out
    waiting on a long page -- this can take 30s-3min or more.

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
    """Search a specific site for pages/listings containing `phrase`, then run
    `analyze_page`-style extraction on each match to answer `question`.

    Works on any site without site-specific scraping code: it uses DuckDuckGo's
    `site:` search to find matches, then trafilatura + Qwen to read and filter
    each page's content, so it keeps working even if the site's layout changes.

    Each match takes roughly 30s-3min to fetch and analyze (more for long pages),
    so a large number of matches will take proportionally long to return.
    Progress is reported per match and per chunk, for MCP clients that respect it.

    Args:
        site: Domain to search within, e.g. "example.com".
        phrase: Exact phrase the publication/listing must contain.
        question: What information to extract from each matching page.
        max_results: Max number of matching pages to analyze. Leave unset (the
            default) to get as many as the search can find -- it aggregates
            across multiple search engines, so this is the real maximum, not
            an arbitrary cap. Set it only to deliberately limit the run.
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
    """List the Qwen/Ollama models currently available on the local Ollama server."""
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
