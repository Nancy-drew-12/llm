
import unittest

import httpx

from unittest.mock import Mock, patch
from openai import APIConnectionError

from day05_schemas import MeaningResult
from day06_guardrails import safe_llm_call
from day06_prompts import SAFE_FALLBACK_RESPONSE


VALID_JSON = """
{
    "summary": "Проблема с аккаунтом",
    "user_goal": "Восстановить доступ",
    "key_details": ["Не удаётся войти"]
}
"""


class TestGuardrails(unittest.TestCase):

    # TEST 1 — EMPTY RESPONSE
    def test_empty_response(self):

        result = safe_llm_call(
            primary_func=lambda: "",
            fallback_func=lambda: "Запасной ответ",
            delay=0,
        )

        self.assertEqual(result, "Запасной ответ")


    # TEST 2 — INVALID JSON
    def test_invalid_json(self):

        result = safe_llm_call(
            primary_func=lambda: "Обычный текст вместо JSON",
            fallback_func=lambda: VALID_JSON,
            schema=MeaningResult,
            delay=0,
        )

        self.assertIsInstance(result, MeaningResult)


    # TEST 3 — MISSING REQUIRED FIELDS
    def test_missing_fields(self):

        result = safe_llm_call(
            primary_func=lambda: '{"summary": "Ошибка"}',
            fallback_func=lambda: VALID_JSON,
            schema=MeaningResult,
            delay=0,
        )

        self.assertIsInstance(result, MeaningResult)


    # TEST 4 — RESPONSE TOO LONG
    def test_response_too_long(self):

        result = safe_llm_call(
            primary_func=lambda: "A" * 5000,
            fallback_func=lambda: "Короткий ответ",
            max_length=1500,
            delay=0,
        )

        self.assertEqual(result, "Короткий ответ")


    # TEST 5 — TEMPORARY API ERROR
    def test_api_retry(self):

        api_call = Mock(
            side_effect=[
                APIConnectionError(
                    request=httpx.Request(
                        "POST",
                        "http://localhost:11434",
                    )
                ),
                "Успешный ответ",
            ]
        )

        result = safe_llm_call(
            primary_func=api_call,
            max_attempts=3,
            delay=0,
        )

        self.assertEqual(result, "Успешный ответ")
        self.assertEqual(api_call.call_count, 2)


    # TEST 6 — ALL API ATTEMPTS FAILED
    def test_all_attempts_failed(self):

        api_call = Mock(
            side_effect=TimeoutError("API unavailable")
        )

        result = safe_llm_call(
            primary_func=api_call,
            max_attempts=3,
            delay=0,
        )

        self.assertIsNone(result)
        self.assertEqual(api_call.call_count, 3)


    # TEST 7 — DEGRADED PIPELINE
    def test_degraded_mode(self):

        from day06 import run_pipeline

        with patch(
            "day06.extract_meaning",
            return_value=None,
        ):
            with patch(
                "day06.client.chat",
                return_value="Ответ без анализа",
            ):
                result = run_pipeline(
                    "Не могу войти в аккаунт"
                )

        self.assertEqual(result, "Ответ без анализа")


    # TEST 8 — SAFE FALLBACK
    def test_safe_fallback(self):

        from day06 import generate_answer

        with patch(
            "day06.client.chat",
            return_value="",
        ):
            result = generate_answer(
                "Помогите войти в аккаунт",
                None,
            )

        self.assertEqual(
            result,
            SAFE_FALLBACK_RESPONSE,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
