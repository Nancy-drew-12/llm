"""День 5: пять последовательных LLM-этапов и семь тестовых обращений.

Зависимости от дня 4 и LLMClient сохранены. Проверь соответствие интерфейса
LLMClient.chat(...) и полей IntentClassification своей версии проекта.
"""

from dotenv import load_dotenv

from llm_client import LLMClient
from day04_schemas import IntentClassification
from day04_prompts import CLASSIFIER_SYSTEM
from router import select_response_prompt
from day05_schemas import MeaningResult, StructuredFields, SelfCheckResult
from day05_prompts import (
    EXTRACT_MEANING_SYSTEM,
    BUILD_FIELDS_SYSTEM,
    GENERATE_ANSWER_SYSTEM,
    SELF_CHECK_SYSTEM,
)

load_dotenv()
client = LLMClient()


def extract_meaning(text: str) -> MeaningResult:
    raw = client.chat(
        system_prompt=EXTRACT_MEANING_SYSTEM,
        user_prompt=text,
        temperature=0,
        json_mode=True,
    )
    return MeaningResult.model_validate_json(raw)


def classify_request(text: str, meaning: MeaningResult) -> IntentClassification:
    raw = client.chat(
        system_prompt=CLASSIFIER_SYSTEM,
        user_prompt=(
            f"Исходное сообщение:\n{text}\n\n"
            f"Анализ смысла:\n{meaning.model_dump_json(indent=2)}"
        ),
        temperature=0,
        json_mode=True,
    )
    return IntentClassification.model_validate_json(raw)


def build_structured_fields(
    text: str,
    meaning: MeaningResult,
    classification: IntentClassification,
) -> StructuredFields:
    raw = client.chat(
        system_prompt=BUILD_FIELDS_SYSTEM,
        user_prompt=(
            f"Исходное сообщение:\n{text}\n\n"
            f"Анализ смысла:\n{meaning.model_dump_json(indent=2)}\n\n"
            f"Классификация:\n{classification.model_dump_json(indent=2)}"
        ),
        temperature=0,
        json_mode=True,
    )
    return StructuredFields.model_validate_json(raw)


def generate_final_answer(
    text: str,
    meaning: MeaningResult,
    classification: IntentClassification,
    fields: StructuredFields,
) -> str:
    instruction = select_response_prompt(classification.category)
    return client.chat(
        system_prompt=GENERATE_ANSWER_SYSTEM,
        user_prompt=(
            f"Исходное сообщение:\n{text}\n\n"
            f"Анализ смысла:\n{meaning.model_dump_json(indent=2)}\n\n"
            f"Классификация:\n{classification.model_dump_json(indent=2)}\n\n"
            f"Поля:\n{fields.model_dump_json(indent=2)}\n\n"
            f"Инструкция категории:\n{instruction}"
        ),
        temperature=0.3,
    )


def self_check(
    text: str,
    meaning: MeaningResult,
    classification: IntentClassification,
    fields: StructuredFields,
    answer: str,
) -> SelfCheckResult:
    raw = client.chat(
        system_prompt=SELF_CHECK_SYSTEM,
        user_prompt=(
            f"Исходное сообщение:\n{text}\n\n"
            f"Анализ смысла:\n{meaning.model_dump_json(indent=2)}\n\n"
            f"Классификация:\n{classification.model_dump_json(indent=2)}\n\n"
            f"Поля:\n{fields.model_dump_json(indent=2)}\n\n"
            f"Итоговый ответ:\n{answer}"
        ),
        temperature=0,
        json_mode=True,
    )
    return SelfCheckResult.model_validate_json(raw)


def run_chain(text: str) -> SelfCheckResult:
    print("\n[INPUT]", text)
    meaning = extract_meaning(text)
    print("\n[STEP 1: EXTRACT MEANING]\n", meaning.model_dump_json(indent=2))

    classification = classify_request(text, meaning)
    print("\n[STEP 2: CLASSIFY]\n", classification.model_dump_json(indent=2))

    fields = build_structured_fields(text, meaning, classification)
    print("\n[STEP 3: STRUCTURED FIELDS]\n", fields.model_dump_json(indent=2))

    answer = generate_final_answer(text, meaning, classification, fields)
    print("\n[STEP 4: GENERATE ANSWER]\n", answer)

    check = self_check(text, meaning, classification, fields, answer)
    print("\n[STEP 5: SELF-CHECK]\n", check.model_dump_json(indent=2))
    print("\n[RESULT]", "PASS" if check.is_valid else "NEEDS REVIEW")
    return check


TEST_CASES = [
    "Уже неделю не могу войти в аккаунт, а поддержка молчит!",
    "Я оплатил подписку вчера, но доступ к функциям так и не появился.",
    "Ваше приложение постоянно зависает. Это ужасно раздражает!",
    "Было бы здорово добавить тёмную тему и возможность менять размер шрифта.",
    "Какие тарифы у вас есть и предусмотрена ли годовая подписка?",
    "Как изменить адрес электронной почты в настройках профиля?",
    "После последнего обновления исчезла история сообщений. Помогите её восстановить.",
]


def main() -> None:
    results = []
    for number, text in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {number}/{len(TEST_CASES)}")
        print("=" * 70)
        try:
            check = run_chain(text)
            status = "PASS" if check.is_valid else "NEEDS REVIEW"
        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")
            status = "ERROR"
        results.append((number, status))

    print("\n" + "=" * 70 + "\nFINAL SUMMARY\n" + "=" * 70)
    for number, status in results:
        print(f"Case {number}: {status}")
    passed = sum(status == "PASS" for _, status in results)
    print(f"\nSelf-check passed: {passed}/{len(TEST_CASES)}")


if __name__ == "__main__":
    main()