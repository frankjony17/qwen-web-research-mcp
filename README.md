# web-research-mcp

MCP server for web research: given a URL (or a site + phrase to search), it fetches the full page content, and uses a local LLM (via [Ollama](https://ollama.com) — any model you have pulled, not tied to a specific model family) to extract/answer a question about it. Long pages are automatically split into chunks and analyzed with a map-reduce strategy, so nothing is skipped.

Content extraction uses [trafilatura](https://trafilatura.readthedocs.io/), which strips navigation/ads/boilerplate based on content heuristics rather than site-specific CSS selectors — so it keeps working if a site's layout changes. Site-scoped search uses DuckDuckGo's `site:` operator, so any domain works without custom scraping code per site.

## Tools

- **`analyze_page(url, question)`** — fetch a page and answer `question` about its full content.
- **`search_site_and_analyze(site, phrase, question, max_results)`** — find pages on `site` containing `phrase`, then run `analyze_page`-style extraction on each match.
- **`list_available_models()`** — list the models available on the configured Ollama server.

## Requirements

- Python 3.11+
- An [Ollama](https://ollama.com) server with at least one model pulled (defaults to `qwen3:14b`, but any Ollama model works)

## Configuration (environment variables)

| Variable         | Default       | Description                                                                                                                                        |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OLLAMA_MODEL`   | `qwen3:14b`   | Ollama model to use                                                                                                                                |
| `OLLAMA_PORT`    | `11434`       | Port of the Ollama server                                                                                                                          |
| `OLLAMA_HOST_IP` | auto-detected | Override the Ollama host IP. If unset, it's read from `ip route show default` (useful when running inside WSL and Ollama runs on the Windows host) |
| `MCP_TRANSPORT`  | `stdio`       | `stdio`, `sse`, or `streamable-http`                                                                                                               |
| `MCP_HOST`       | `127.0.0.1`   | Bind host (HTTP transports only)                                                                                                                   |
| `MCP_PORT`       | `8000`        | Bind port (HTTP transports only)                                                                                                                   |

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m web_research.server
```

## Run with Docker

```bash
docker build -t web-research-mcp .
docker run -p 8001:8000 \
  -e MCP_TRANSPORT=streamable-http \
  -e MCP_HOST=0.0.0.0 \
  -e OLLAMA_HOST_IP=host.docker.internal \
  web-research-mcp
```

Or as a service in a `docker-compose.yml` alongside other tools (e.g. [Open WebUI](https://github.com/open-webui/open-webui)):

```yaml
web-research:
  build: ./web-research-mcp
  ports:
    - "8001:8000"
  extra_hosts:
    - "host.docker.internal:host-gateway"
  environment:
    - MCP_TRANSPORT=streamable-http
    - MCP_HOST=0.0.0.0
    - MCP_PORT=8000
    - OLLAMA_HOST_IP=host.docker.internal
    - OLLAMA_MODEL=qwen3:14b
```

Then point your MCP client (e.g. Open WebUI's Tools/Connections settings) at `http://web-research:8000/mcp` (internal Docker network) or `http://localhost:8001/mcp` (from the host).
