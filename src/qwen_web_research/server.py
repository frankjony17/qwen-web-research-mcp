"""MCP server exposing web scraping + Qwen-powered analysis as tools."""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from . import ollama_client
from .analysis import analyze_text
from .scraper import FetchError, fetch_page_text
from .search import search_site

mcp = MCPServer("qwen-web-research")


@mcp.tool()
def analyze_page(url: str, question: str) -> str:
    """Fetch a web page, and use a local Qwen model to extract/answer `question`
    about its full content. Long pages are automatically split into chunks and
    analyzed piece by piece (map-reduce), so no content is skipped.

    Args:
        url: The page to fetch and analyze.
        question: What to extract or answer about the page's content.
    """
    try:
        page = fetch_page_text(url)
    except FetchError as exc:
        return f"Error: {exc}"

    answer = analyze_text(page["text"], question)
    title = page["title"] or url
    return f"# {title}\n{url}\n\n{answer}"


@mcp.tool()
def search_site_and_analyze(site: str, phrase: str, question: str, max_results: int = 5) -> str:
    """Search a specific site for pages/listings containing `phrase`, then run
    `analyze_page`-style extraction on each match to answer `question`.

    Works on any site without site-specific scraping code: it uses DuckDuckGo's
    `site:` search to find matches, then trafilatura + Qwen to read and filter
    each page's content, so it keeps working even if the site's layout changes.

    Each match takes roughly 30s-3min to fetch and analyze (more for long pages),
    so a large max_results will take proportionally long to return.

    Args:
        site: Domain to search within, e.g. "example.com".
        phrase: Exact phrase the publication/listing must contain.
        question: What information to extract from each matching page.
        max_results: Max number of matching pages to analyze (default 5).
    """
    matches = search_site(site, phrase, max_results=max_results)
    if not matches:
        return f'No se encontraron resultados en {site} para la frase "{phrase}".'

    sections = []
    for match in matches:
        url = match["url"]
        if not url:
            continue
        try:
            page = fetch_page_text(url)
            answer = analyze_text(page["text"], question)
            title = page["title"] or match["title"] or url
            sections.append(f"## {title}\n{url}\n\n{answer}")
        except FetchError as exc:
            sections.append(f"## {match.get('title') or url}\n{url}\n\nError: {exc}")

    return "\n\n---\n\n".join(sections)


@mcp.tool()
def list_available_models() -> list[str]:
    """List the Qwen/Ollama models currently available on the local Ollama server."""
    return ollama_client.list_models()


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
