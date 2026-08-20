"""Fetch a URL and extract its main readable content, regardless of site layout."""
from __future__ import annotations

import asyncio
import uuid

import httpx
import trafilatura
from mcp.server.mcpserver import Context
from pydantic import BaseModel, Field

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 qwen-web-research-mcp/0.1"
)

# Only timeouts are retried (transient) -- a 404 or DNS failure won't fix itself.
MAX_TIMEOUT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0

# How many times we ask "did you finish the human check?" before giving up.
MAX_VERIFICATION_ROUNDS = 3

# Rough signals for "this is a bot-verification page", not real content.
# We only ask the human to solve it themselves -- never attempted programmatically.
CHALLENGE_MARKERS = (
    "checking your browser",
    "verifying you are human",
    "verify you are human",
    "just a moment",
    "attention required! | cloudflare",
    "cf-turnstile",
    "g-recaptcha",
    "hcaptcha",
    "please enable javascript and cookies",
    "ddos protection by",
)


class VerificationDone(BaseModel):
    done: bool = Field(description="True once you've completed the check in your browser and want to retry.")


class FetchError(RuntimeError):
    pass


def _looks_like_challenge_page(response: httpx.Response) -> bool:
    if response.status_code in (403, 503):
        return True
    lowered = response.text[:4000].lower()
    return any(marker in lowered for marker in CHALLENGE_MARKERS)


async def _handle_human_verification(url: str, ctx: Context | None) -> bool:
    """Ask the human to solve the check themselves. Returns True if they say they did."""
    if ctx is None:
        return False

    url_result = await ctx.elicit_url(
        message=f"La página {url} pide verificación humana (Cloudflare/CAPTCHA). Ábrela y resuélvela.",
        url=url,
        elicitation_id=str(uuid.uuid4()),
    )
    if url_result.action != "accept":
        return False

    for _ in range(MAX_VERIFICATION_ROUNDS):
        result = await ctx.elicit(
            "¿Ya completaste la verificación en el navegador? Confirma para reintentar.",
            VerificationDone,
        )
        if result.action == "accept" and result.data.done:
            return True
        if result.action != "accept":
            return False
    return False


async def fetch_page_text(url: str, *, timeout: float = 20.0, ctx: Context | None = None) -> dict:
    """Download a URL and extract clean article text + metadata.

    Uses trafilatura's extraction, which strips navigation/ads/boilerplate
    based on content heuristics rather than site-specific CSS selectors, so
    it keeps working if a site's markup changes.

    If the page turns out to be a bot-verification challenge (Cloudflare,
    CAPTCHA, etc.), this asks the human user -- via MCP elicitation -- to solve
    it themselves in their own browser, then retries. It never attempts to
    solve or bypass the check programmatically.
    """
    response = None
    last_timeout_exc: httpx.TimeoutException | None = None
    verification_attempted = False
    async with httpx.AsyncClient() as client:
        attempt = 0
        while attempt <= MAX_TIMEOUT_RETRIES:
            try:
                response = await client.get(
                    url,
                    timeout=timeout,
                    follow_redirects=True,
                    headers={"User-Agent": USER_AGENT},
                )
            except httpx.TimeoutException as exc:
                last_timeout_exc = exc
                if attempt < MAX_TIMEOUT_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    if ctx:
                        await ctx.report_progress(
                            attempt + 1,
                            MAX_TIMEOUT_RETRIES + 1,
                            f"Timeout obteniendo {url}, reintentando en {wait:.0f}s...",
                        )
                    await asyncio.sleep(wait)
                    attempt += 1
                    continue
                raise FetchError(
                    f"Failed to fetch {url}: timed out after {MAX_TIMEOUT_RETRIES + 1} attempts"
                ) from last_timeout_exc
            except httpx.HTTPError as exc:
                raise FetchError(f"Failed to fetch {url}: {exc}") from exc

            if _looks_like_challenge_page(response):
                if not verification_attempted:
                    verification_attempted = True
                    if await _handle_human_verification(url, ctx):
                        continue  # retry the fetch without counting against attempt
                raise FetchError(
                    f"{url} requiere verificación humana (Cloudflare/CAPTCHA) y no se pudo confirmar "
                    "que se haya resuelto."
                )

            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise FetchError(f"Failed to fetch {url}: {exc}") from exc
            break

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
