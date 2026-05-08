import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph import make_graph
from state import UserState
from tracing import init_tracing

_root = Path(__file__).resolve().parent
if not load_dotenv(_root / ".env"):
    load_dotenv(_root / "env_vars.env")

init_tracing()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "Задайте OPENAI_API_KEY в файле .env или env_vars.env в корне проекта "
        "(строка вида OPENAI_API_KEY=sk-...)."
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=_api_key,
    temperature=0,
    max_retries=3,
    timeout=240,
)

state: UserState = {"query": ""}
graph = make_graph(state, llm)


print("Multi-Agent AI Nutritionist")
print("Пиши запросы. Примеры:")
print("- Сохрани профиль: рост 178, вес 70, возраст 20, мужчина, активность средняя, цель набор веса")
print("- Рассчитай цели")
print("- Составь меню на неделю без курицы, 3 приёма пищи")

while True:
    try:
        q = input("> ").strip()
    except KeyboardInterrupt:
        break
    if not q:
        continue
    state = {"query": q}
    out = graph.invoke(state)
    print("\n" + (out.get("final_answer") or "(no output)") + "\n")
