# Elbrus Holiday Task: LangGraph

## Полезные материалы

Прочитайте перед тем как начинать задания — займёт 20-30 минут и сильно упростит понимание.

| | Статья | Что даёт |
|---|---|---|
| 1 | [Введение в LangGraph (Habr)](https://habr.com/ru/companies/amvera/articles/933460/) | Базовые концепции: StateGraph, nodes, edges, checkpointer — с примерами на русском |
| 2 | [LangChain vs LangGraph (Habr)](https://habr.com/ru/articles/956940/) | Объясняет, зачем вообще LangGraph, если уже есть LangChain — хорошо читать после первой |

---

## Контекст

Вы уже строили LCEL-пайплайны, подключали ретриверы, работали с памятью через `session_store`, писали агентные циклы с инструментами.

Теперь LangGraph. Главная идея — вместо цепочки вы строите **граф**: узлы (nodes) — шаги обработки, рёбра (edges) — переходы между ними, а **State** — объект данных, который путешествует через весь граф.

Три понятия, которые нужно усвоить:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):       # 1. State — что передаётся между узлами
    question: str
    answer: str

def my_node(state: State) -> dict:   # 2. Node — функция, возвращает обновление стейта
    return {"answer": "привет"}

graph = StateGraph(State)
graph.add_node("node", my_node)      # 3. Edge — соединение между узлами
graph.set_entry_point("node")
graph.add_edge("node", END)

app = graph.compile()
app.invoke({"question": "что-то", "answer": ""})
```

---

## Задание 1: Простейший граф (HR-ассистент)

Соберите граф из трёх узлов на основе кода из лекций. Никакого нового функционала — только перенос логики в граф.

### Стейт

```python
class HRState(TypedDict):
    question: str
    route: str        # "search" | "chat"
    documents: list
    answer: str
```

### Узлы

**`classify_node`** — вызывает LLM, определяет маршрут. Заполняет поле `route`.

**`search_node`** — ищет вакансии в Qdrant. Заполняет поле `documents`.

**`answer_node`** — генерирует ответ. Если `documents` не пустой — использует их как контекст, если пустой — просто отвечает.

### Граф

```
classify_node → (если "search") → search_node → answer_node → END
              → (если "chat")   → answer_node → END
```

Условный переход реализуется через `add_conditional_edges`:

```python
def route_fn(state: HRState) -> str:
    return state["route"]

graph.add_conditional_edges("classify", route_fn, {
    "search": "search",
    "chat": "answer"
})
```

### Что сделать

1. Собрать граф, скомпилировать, запустить.
2. Визуализировать:
   ```python
   from IPython.display import Image
   Image(app.get_graph().draw_mermaid_png())
   ```
3. Проверить три запроса:
   - «Найди вакансии Python-разработчика»
   - «Привет, как дела?»
   - «Сколько платят ML-инженерам?»
4. Распечатать финальный стейт после каждого вызова.

---

## Задание 2: Персистентная память

В предыдущей теме вы хранили историю диалога в `session_store` — словаре, который вы создавали сами и сами передавали в каждый вызов.

В LangGraph память встроена в граф через **checkpointer**. Граф сам сохраняет весь стейт после каждого шага, и при следующем вызове с тем же `thread_id` — продолжает с того места.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session_1"}}

app.invoke({"question": "найди Python вакансии", ...}, config=config)
app.invoke({"question": "а есть удалёнка среди них?", ...}, config=config)
```

### Что сделать

1. Добавьте в `HRState` поле `history: list` — список пар вопрос/ответ.

2. Обновите `answer_node`: после генерации ответа добавляйте в `history` текущую пару и передавайте историю в промпт LLM при следующем вызове.

3. Подключите `MemorySaver` к графу из Задания 1.

4. Проведите диалог в трёх `invoke` с одним `thread_id`:
   ```
   → «Найди вакансии Python-разработчика в Питере»
   → «Какие из них с удалёнкой?»
   → «Покажи требования к первой подробнее»
   ```

5. Повторите с **другим** `thread_id` — убедитесь, что это новый диалог.

6. После третьего сообщения вызовите `app.get_state(config)` и посмотрите, что хранится.

---

## Задание 3: Апгрейд своего проекта

Возьмите свой проект (book / dish / movie) и перенесите его на LangGraph.

### Минимальный апгрейд

1. Опишите `State` — что должен помнить граф? (показанные результаты, предпочтения пользователя, история)
2. Разбейте логику на узлы (минимум 3).
3. Подключите `MemorySaver`.
4. Покажите диалог на 3 хода, где второй вопрос опирается на первый.

### Один дополнительный сценарий на выбор

**A. Retry при пустом результате** — если поиск вернул пустой список, граф перефразирует запрос через LLM и пробует ещё раз (но не более 2 попыток).

**B. Human-in-the-loop** — если запрос слишком общий, граф останавливается и ждёт уточнения:
```python
app = graph.compile(checkpointer=memory, interrupt_before=["search"])
# после остановки:
app.update_state(config, {"question": "уточнённый запрос"})
app.invoke(None, config)  # продолжить
```

**C. Персонализация** — граф запоминает уже показанные результаты в `shown_items` и при следующем поиске говорит LLM: «уже предлагали X, предложи другое».

---

## Критерии сдачи

- Задания 1 и 2: граф запускается, визуализация прилагается, после каждого задания — пара предложений: что заметили, что было неочевидно
- Задание 3: рабочий граф с памятью + диалог на 3 хода + один из сценариев A/B/C
