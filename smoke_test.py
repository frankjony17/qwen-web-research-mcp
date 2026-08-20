import sys
sys.path.insert(0, "src")

from qwen_web_research.ollama_client import list_models, ollama_base_url
from qwen_web_research.scraper import fetch_page_text
from qwen_web_research.analysis import analyze_text

print("Ollama base URL:", ollama_base_url())
print("Models:", list_models())

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
print(f"\nFetching {url} ...")
page = fetch_page_text(url)
print("Title:", page["title"])
print("Text length:", len(page["text"]), "chars")

print("\nAsking Qwen a question about it...")
answer = analyze_text(page["text"], "Who created Python and in what year was it first released?")
print("\n--- Answer ---")
print(answer)
