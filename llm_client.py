"""Клиент для вызова LLM. Здесь только обращение к API, без логики пайплайна.

Работает с любым OpenAI-совместимым API:
- OpenAI:      OPENAI_API_KEY=sk-...           (OPENAI_BASE_URL не нужен)
- OpenRouter:  OPENAI_BASE_URL=https://openrouter.ai/api/v1
- Ollama:      OPENAI_BASE_URL=http://localhost:11434/v1, OPENAI_API_KEY=ollama
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model=None, api_key=None, base_url=None, timeout=60):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Не найден OPENAI_API_KEY. Скопируй .env.example в .env и впиши ключ.")

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )

    def chat(self, system_prompt, user_prompt, temperature=0.2, max_tokens=800, json_mode=False):
        """Отправляет запрос и возвращает текст ответа модели."""
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        logger.info("Запрос к модели %s (json_mode=%s)", self.model, json_mode)
        response = self.client.chat.completions.create(**params)

        content = response.choices[0].message.content or ""
        logger.info("Ответ получен: %d символов", len(content))
        return content
