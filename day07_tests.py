
import json
import unittest

from unittest.mock import patch
from pathlib import Path

from day07 import run_pipeline
from day06_prompts import SAFE_FALLBACK_RESPONSE

from day04_schemas import IntentClassification

from day05_schemas import (
    MeaningResult,
    StructuredFields,
    SelfCheckResult,
)


MEANING = MeaningResult(
    summary="Проблема с входом в аккаунт",
    user_goal="Восстановить доступ",
    key_details=["Не удаётся войти"],
)

CLASSIFICATION = IntentClassification(
    category="support",
    intent="login_issue",
)

FIELDS = StructuredFields(
    problem="Не удаётся войти в аккаунт",
    duration=None,
    requested_action="Восстановить доступ",
    important_facts=["Проблема с входом"],
)

CHECK = SelfCheckResult(
    is_valid=True,
    contradictions=[],
    missing_details=[],
    unsupported_claims=[],
    explanation="Ответ корректен",
)

ANSWER = "Попробуйте восстановить пароль."


class TestFinalPipeline(unittest.TestCase):

    # TEST 1 — FULL PIPELINE
    def test_successful_pipeline(self):

        with patch(
            "day07.safe_json_request",
            side_effect=[
                MEANING,
                CLASSIFICATION,
                FIELDS,
                CHECK,
            ],
        ):
            with patch(
                "day07.safe_text_request",
                return_value=ANSWER,
            ):
                result = run_pipeline(
                    "Не могу войти в аккаунт"
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], ANSWER)
        self.assertEqual(result["failed_stages"], [])
        self.assertEqual(
            result["classification"]["category"],
            "support",
        )


    # TEST 2 — EMPTY INPUT
    def test_empty_input(self):

        result = run_pipeline("   ")

        self.assertEqual(
            result["status"],
            "invalid_input",
        )


    # TEST 3 — PARTIAL FAILURE
    def test_degraded_pipeline(self):

        with patch(
            "day07.safe_json_request",
            side_effect=[
                None,
                CLASSIFICATION,
                FIELDS,
                CHECK,
            ],
        ):
            with patch(
                "day07.safe_text_request",
                return_value=ANSWER,
            ):
                result = run_pipeline(
                    "Не могу войти в аккаунт"
                )

        self.assertEqual(result["status"], "degraded")
        self.assertIn(
            "meaning",
            result["failed_stages"],
        )
        self.assertEqual(result["answer"], ANSWER)


    # TEST 4 — SAFE FALLBACK
    def test_safe_fallback(self):

        with patch(
            "day07.safe_json_request",
            return_value=None,
        ):
            with patch(
                "day07.safe_text_request",
                return_value=None,
            ):
                result = run_pipeline(
                    "Помогите войти в аккаунт"
                )

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(
            result["answer"],
            SAFE_FALLBACK_RESPONSE,
        )


    # TEST 5 — SELF-CHECK REJECTED ANSWER
    def test_needs_review(self):

        failed_check = SelfCheckResult(
            is_valid=False,
            contradictions=[],
            missing_details=[],
            unsupported_claims=[
                "Неподтверждённое обещание"
            ],
            explanation="Ответ требует проверки",
        )

        with patch(
            "day07.safe_json_request",
            side_effect=[
                MEANING,
                CLASSIFICATION,
                FIELDS,
                failed_check,
            ],
        ):
            with patch(
                "day07.safe_text_request",
                return_value="Мы восстановим аккаунт.",
            ):
                result = run_pipeline(
                    "Не могу войти в аккаунт"
                )

        self.assertEqual(
            result["status"],
            "needs_review",
        )


    # TEST 6 — DEMO INPUTS
    def test_demo_inputs(self):

        path = (
            Path(__file__).resolve().parent
            / "demo"
            / "inputs.json"
        )

        with path.open("r", encoding="utf-8") as file:
            examples = json.load(file)

        self.assertIsInstance(examples, list)
        self.assertGreaterEqual(len(examples), 5)

        for example in examples:
            self.assertIsInstance(example, str)
            self.assertTrue(example.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
