"""Split long text into overlapping chunks sized for the model's context window."""
from __future__ import annotations

# Rough estimate: ~4 chars per token.
CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_TOKENS = 3000
DEFAULT_OVERLAP_TOKENS = 150


def split_into_chunks(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    chunk_size = chunk_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
