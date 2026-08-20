"""Map-reduce analysis of long page content with Qwen."""
from __future__ import annotations

from . import ollama_client
from .chunking import split_into_chunks

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


def analyze_text(text: str, question: str, *, model: str = ollama_client.DEFAULT_MODEL) -> str:
    """Answer `question` about `text`, chunking + map-reduce if the text is long."""
    chunks = split_into_chunks(text)

    if len(chunks) == 1:
        return ollama_client.chat(
            f"Pregunta/filtro: {question}\n\nContenido:\n{chunks[0]}",
            system=MAP_SYSTEM_PROMPT,
            model=model,
        )

    partial_results = []
    for i, chunk in enumerate(chunks, start=1):
        result = ollama_client.chat(
            f"Pregunta/filtro: {question}\n\nFragmento {i}/{len(chunks)}:\n{chunk}",
            system=MAP_SYSTEM_PROMPT,
            model=model,
        )
        partial_results.append(f"--- Extracto {i} ---\n{result}")

    combined = "\n\n".join(partial_results)
    return ollama_client.chat(
        f"Pregunta original: {question}\n\nExtractos parciales:\n{combined}",
        system=REDUCE_SYSTEM_PROMPT,
        model=model,
    )
