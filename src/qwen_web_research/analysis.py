"""Map-reduce analysis of long page content with Qwen."""
from __future__ import annotations

import logging

import httpx
from mcp.server.mcpserver import Context

from . import ollama_client
from .chunking import split_into_chunks

logger = logging.getLogger("qwen_web_research.analysis")

# Caps how long a huge page can take to process.
MAX_CHUNKS = 12

MAP_SYSTEM_PROMPT = (
    "Eres un asistente de investigación. Se te da un fragmento de una página web "
    "y una pregunta/filtro. Extrae ÚNICAMENTE la información del fragmento que sea "
    "relevante para la pregunta. Si el fragmento no contiene nada relevante, responde "
    "exactamente: 'Sin información relevante.' No inventes información que no esté en el texto."
)

REDUCE_SYSTEM_PROMPT = (
    "Eres un asistente de investigación. Se te dan varios extractos parciales, "
    "obtenidos de distintas secciones de una misma página web, cada uno ya filtrado "
    "por relevancia a una pregunta. Combínalos en una respuesta final única, clara y "
    "sin repeticiones, que responda la pregunta original. Ignora los extractos que digan "
    "'Sin información relevante.' Si ningún extracto tiene información útil, dilo explícitamente."
)


async def analyze_text(
    text: str,
    question: str,
    *,
    model: str = ollama_client.DEFAULT_MODEL,
    ctx: Context | None = None,
) -> str:
    """Answer `question` about `text`, chunking + map-reduce if the text is long."""
    chunks = split_into_chunks(text)
    logger.info("analyze_text start chars=%d chunks=%d model=%s", len(text), len(chunks), model)

    if len(chunks) == 1:
        if ctx:
            await ctx.report_progress(0, 1, "Analizando página...")
        try:
            result = await ollama_client.chat(
                f"Pregunta/filtro: {question}\n\nContenido:\n{chunks[0]}",
                system=MAP_SYSTEM_PROMPT,
                model=model,
            )
            logger.info("analyze_text done (single chunk) answer_chars=%d", len(result))
            return result
        except httpx.TimeoutException:
            logger.error("analyze_text timeout (single chunk)")
            return "Error: el modelo tardó demasiado en responder (timeout)."

    truncated = len(chunks) > MAX_CHUNKS
    chunks = chunks[:MAX_CHUNKS]
    if truncated:
        logger.warning("analyze_text truncating to first %d chunks (page was longer)", MAX_CHUNKS)

    partial_results = []
    for i, chunk in enumerate(chunks, start=1):
        if ctx:
            await ctx.report_progress(i - 1, len(chunks) + 1, f"Analizando fragmento {i}/{len(chunks)}...")
        logger.info("analyze_text chunk %d/%d start chars=%d", i, len(chunks), len(chunk))
        try:
            result = await ollama_client.chat(
                f"Pregunta/filtro: {question}\n\nFragmento {i}/{len(chunks)}:\n{chunk}",
                system=MAP_SYSTEM_PROMPT,
                model=model,
            )
            logger.info("analyze_text chunk %d/%d done", i, len(chunks))
        except httpx.TimeoutException:
            logger.error("analyze_text chunk %d/%d timeout, skipping", i, len(chunks))
            result = "Error: timeout analizando este fragmento, se omitió."
        partial_results.append(f"--- Extracto {i} ---\n{result}")

    if truncated:
        partial_results.append(
            f"(Nota: la página es muy larga, solo se analizaron los primeros {MAX_CHUNKS} fragmentos.)"
        )

    if ctx:
        await ctx.report_progress(len(chunks), len(chunks) + 1, "Combinando resultados...")

    logger.info("analyze_text reduce start (%d partial results)", len(partial_results))
    combined = "\n\n".join(partial_results)
    try:
        result = await ollama_client.chat(
            f"Pregunta original: {question}\n\nExtractos parciales:\n{combined}",
            system=REDUCE_SYSTEM_PROMPT,
            model=model,
        )
        logger.info("analyze_text done answer_chars=%d", len(result))
        return result
    except httpx.TimeoutException:
        logger.error("analyze_text reduce timeout")
        return "Error: timeout combinando los extractos. Extractos parciales:\n\n" + combined
