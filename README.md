# Debitor Voice Agent

Стартовый каркас для голосового ассистента на базе [LangGraph](https://github.com/langchain-ai/langgraph).
Архитектура изначально ориентирована на сложные нелинейные переходы и
дальнейшее масштабирование.

## Структура проекта

```
src/debitor/
    settings.py        # конфигурация через pydantic
    main.py            # пример запуска графа
    graph/
        state.py       # описание состояния
        nodes/         # обработчики узлов
            intent.py
            pay.py
            balance.py
            fallback.py
        workflow.py    # сборка графа
tests/                 # модульные тесты
```

## Работа с зависимостями

Проект использует [uv](https://github.com/astral-sh/uv) как пакетный менеджер.

```bash
# установка зависимостей
uv sync --all-extras --group dev

# запуск тестов
uv run pytest

# запуск примера
PYTHONPATH=src uv run python src/debitor/main.py
```
