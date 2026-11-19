"""День 3: схема structured output и разбор ответа модели.

parse_analysis(raw) превращает сырой текст модели в объект TextAnalysis
или бросает LLMOutputError с понятным сообщением: что именно сломалось.
"""

import json
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from prompts import KEY_POINTS_COUNT


class Category(str, Enum):
    complaint = "complaint"
    question = "question"
    request = "request"
    news = "news"
    other = "other"


class Sentiment(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class TextAnalysis(BaseModel):
    """Ожидаемый ответ модели. Структура и типы проверяются строго."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: Category
    sentiment: Sentiment
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=KEY_POINTS_COUNT, max_length=KEY_POINTS_COUNT)
    final_answer: str = Field(min_length=1)

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def lowercase_label(cls, value):
        # "Complaint " -> "complaint": регистр и пробелы — не повод отклонять ответ
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("key_points")
    @classmethod
    def no_empty_points(cls, points):
        if any(not p.strip() for p in points):
            raise ValueError("в key_points есть пустой пункт")
        return points


class LLMOutputError(Exception):
    """Ответ модели нельзя использовать. kind: empty | invalid_json | not_object | schema."""

    def __init__(self, kind, message, raw=""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.raw = raw


def _extract_json_text(raw):
    """Убирает обёртку ```json ... ``` и текст вокруг, если модель их добавила."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0) if match else raw


def _explain_validation_error(error: ValidationError):
    """Переводит ошибки pydantic в понятные строки по-русски."""
    allowed = {
        "category": ", ".join(c.value for c in Category),
        "sentiment": ", ".join(s.value for s in Sentiment),
    }
    lines = []
    for e in error.errors():
        field = ".".join(str(part) for part in e["loc"]) or "ответ"
        kind = e["type"]
        got = e.get("input")
        if kind == "missing":
            lines.append(f"нет обязательного поля «{field}»")
        elif kind == "enum":
            lines.append(f"поле «{field}»: недопустимое значение {got!r}, ожидалось одно из: {allowed.get(field, '?')}")
        elif kind == "string_type":
            lines.append(f"поле «{field}»: ожидалась строка, получено {type(got).__name__} ({got!r})")
        elif kind == "list_type":
            lines.append(f"поле «{field}»: ожидался список строк, получено {type(got).__name__}")
        elif kind in ("too_short", "too_long") and field == "key_points":
            lines.append(f"поле «key_points»: ожидалось ровно {KEY_POINTS_COUNT} пункта, получено {len(got)}")
        elif kind == "string_too_short":
            lines.append(f"поле «{field}»: пустая строка")
        else:
            lines.append(f"поле «{field}»: {e['msg']}")
    return lines


def parse_analysis(raw):
    """Сырой ответ модели -> TextAnalysis. Любая проблема -> LLMOutputError с объяснением."""
    if raw is None or not raw.strip():
        raise LLMOutputError("empty", "Модель вернула пустой ответ", raw or "")

    text = _extract_json_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[max(0, e.pos - 30): e.pos + 30].replace("\n", " ")
        raise LLMOutputError(
            "invalid_json",
            f"Модель вернула невалидный JSON: {e.msg} (строка {e.lineno}, позиция {e.colno}). "
            f"Место ошибки: …{snippet}…",
            raw,
        ) from e

    if not isinstance(data, dict):
        raise LLMOutputError("not_object", f"Ожидался JSON-объект {{...}}, получен {type(data).__name__}", raw)

    try:
        return TextAnalysis.model_validate(data)
    except ValidationError as e:
        problems = _explain_validation_error(e)
        raise LLMOutputError("schema", "Ответ не прошёл проверку схемы: " + "; ".join(problems), raw) from e