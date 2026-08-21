import asyncio
import sys

sys.path.insert(0, "src")

from web_research.ollama_client import list_models, ollama_base_url
from web_research.scraper import fetch_page_text
from web_research.analysis import analyze_text


async def main() -> None:
    print("Ollama base URL:", ollama_base_url())
    print("Models:", await list_models())

    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print(f"\nFetching {url} ...")
    page = await fetch_page_text(url)
    print("Title:", page["title"])
    print("Text length:", len(page["text"]), "chars")

    print("\nAsking the model a question about it...")
    answer = await analyze_text(page["text"], "Who created Python and in what year was it first released?")
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
