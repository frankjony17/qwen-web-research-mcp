"""MCP server exposing web scraping + Qwen-powered analysis as tools."""
from __future__ import annotations

import os

from mcp.server.mcpserver import Context, MCPServer

from . import ollama_client
from .analysis import analyze_text
from .scraper import FetchError, fetch_page_text
from .search import search_site

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
    try:
        page = await fetch_page_text(url, ctx=ctx)
    except FetchError as exc:
        return f"Error: {exc}"

    answer = await analyze_text(page["text"], question, ctx=ctx)
    title = page["title"] or url
    return f"# {title}\n{url}\n\n{answer}"


@mcp.tool()
async def search_site_and_analyze(
    site: str, phrase: str, question: str, ctx: Context, max_results: int = 5
) -> str:
    """Search a specific site for pages/listings containing `phrase`, then run
    `analyze_page`-style extraction on each match to answer `question`.

    Works on any site without site-specific scraping code: it uses DuckDuckGo's
    `site:` search to find matches, then trafilatura + Qwen to read and filter
    each page's content, so it keeps working even if the site's layout changes.

    Each match takes roughly 30s-3min to fetch and analyze (more for long pages),
    so a large max_results will take proportionally long to return. Progress is
    reported per match and per chunk, for MCP clients that respect it.

    Args:
        site: Domain to search within, e.g. "example.com".
        phrase: Exact phrase the publication/listing must contain.
        question: What information to extract from each matching page.
        max_results: Max number of matching pages to analyze (default 5).
    """
    matches = await search_site(site, phrase, max_results=max_results, ctx=ctx)
    if not matches:
        return f'No se encontraron resultados en {site} para la frase "{phrase}".'

    sections = []
    for idx, match in enumerate(matches, start=1):
        url = match["url"]
        if not url:
            continue
        await ctx.report_progress(idx - 1, len(matches), f"Procesando resultado {idx}/{len(matches)}: {url}")
        try:
            page = await fetch_page_text(url, ctx=ctx)
            answer = await analyze_text(page["text"], question, ctx=ctx)
            title = page["title"] or match["title"] or url
            sections.append(f"## {title}\n{url}\n\n{answer}")
        except FetchError as exc:
            sections.append(f"## {match.get('title') or url}\n{url}\n\nError: {exc}")

    return "\n\n---\n\n".join(sections)


@mcp.tool()
async def list_available_models() -> list[str]:
    """List the Qwen/Ollama models currently available on the local Ollama server."""
    return await ollama_client.list_models()


def main() -> None:
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
