
from dotenv import load_dotenv

from llm_client import LLMClient
from day04_schemas import IntentClassification
from day04_prompts import CLASSIFIER_SYSTEM
from router import select_response_prompt

from day05_schemas import MeaningResult, StructuredFields
from day05_prompts import (
    EXTRACT_MEANING_SYSTEM,
    BUILD_FIELDS_SYSTEM,
    GENERATE_ANSWER_SYSTEM,
)


load_dotenv()
client = LLMClient()


# STEP 1 — EXTRACT MEANING

def extract_meaning(text: str) -> MeaningResult:
    raw = client.chat(
        system_prompt=EXTRACT_MEANING_SYSTEM,
        user_prompt=text,
        temperature=0,
        json_mode=True,
    )

    return MeaningResult.model_validate_json(raw)


# STEP 2 — CLASSIFY REQUEST

def classify_request(
    text: str,
    meaning: MeaningResult,
) -> IntentClassification:

    classification_input = (
        f"Исходное сообщение:\n{text}\n\n"
        f"Результат анализа смысла:\n"
        f"{meaning.model_dump_json(indent=2)}"
    )

    raw = client.chat(
        system_prompt=CLASSIFIER_SYSTEM,
        user_prompt=classification_input,
        temperature=0,
        json_mode=True,
    )

    return IntentClassification.model_validate_json(raw)


# STEP 3 — BUILD STRUCTURED FIELDS

def build_structured_fields(
    text: str,
    meaning: MeaningResult,
    classification: IntentClassification,
) -> StructuredFields:

    fields_input = (
        f"Исходное сообщение:\n{text}\n\n"
        f"Анализ смысла:\n"
        f"{meaning.model_dump_json(indent=2)}\n\n"
        f"Классификация:\n"
        f"{classification.model_dump_json(indent=2)}"
    )

    raw = client.chat(
        system_prompt=BUILD_FIELDS_SYSTEM,
        user_prompt=fields_input,
        temperature=0,
        json_mode=True,
    )

    return StructuredFields.model_validate_json(raw)


# STEP 4 — GENERATE FINAL ANSWER

def generate_final_answer(
    text: str,
    meaning: MeaningResult,
    classification: IntentClassification,
    fields: StructuredFields,
) -> str:

    instruction = select_response_prompt(
        classification.category
    )

    answer_input = (
        f"Исходное сообщение:\n{text}\n\n"
        f"Анализ смысла:\n"
        f"{meaning.model_dump_json(indent=2)}\n\n"
        f"Классификация:\n"
        f"{classification.model_dump_json(indent=2)}\n\n"
        f"Структурированные поля:\n"
        f"{fields.model_dump_json(indent=2)}\n\n"
        f"Инструкция категории:\n{instruction}"
    )

    response = client.chat(
        system_prompt=GENERATE_ANSWER_SYSTEM,
        user_prompt=answer_input,
        temperature=0.3,
    )

    return response


if __name__ == "__main__":
    text = (
        "Уже неделю не могу войти в аккаунт, "
        "а поддержка молчит!"
    )

    # STEP 1
    meaning = extract_meaning(text)

    print("\n[STEP 1: EXTRACT MEANING]")
    print(meaning.model_dump_json(indent=2))

    # STEP 2
    classification = classify_request(text, meaning)

    print("\n[STEP 2: CLASSIFY REQUEST]")
    print(classification.model_dump_json(indent=2))

    # STEP 3
    fields = build_structured_fields(
        text,
        meaning,
        classification,
    )

    print("\n[STEP 3: BUILD STRUCTURED FIELDS]")
    print(fields.model_dump_json(indent=2))

    # STEP 4
    answer = generate_final_answer(
        text,
        meaning,
        classification,
        fields,
    )

    print("\n[STEP 4: GENERATE FINAL ANSWER]")
    print(answer)
