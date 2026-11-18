# LLM pipeline для обработки текста

Учебный проект: модульный LLM-пайплайн, который принимает текст, извлекает из него структуру и генерирует полезный ответ. Задание курса — в `llm_overall.txt`.

## День 1 — базовый пайплайн

Для каждого текста пайплайн возвращает:
- `summary` — краткое содержание;
- `key_points` — 3 ключевые мысли;
- `response` — короткий полезный ответ автору.

### Структура
```
main.py          # собирает всё вместе: читает тексты, вызывает модель, печатает и сохраняет
llm_client.py    # только вызов LLM API
prompts.py       # шаблоны промптов
sample_inputs/   # тестовые тексты (жалоба, вопрос, новость)
results/         # результаты прогонов (JSON)
```

### Запуск
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # и вписать ключ
python main.py                # все тексты из sample_inputs/
python main.py my_text.txt    # свои файлы
```

Подходит любой OpenAI-совместимый API: OpenAI, OpenRouter, Ollama (локально). Провайдер и модель задаются в `.env`, код менять не нужно.
