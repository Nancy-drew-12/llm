
import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient
from router import select_response_prompt

from day04_schemas import IntentClassification
from day04_prompts import CLASSIFIER_SYSTEM

from day05_schemas import (
    MeaningResult,
    StructuredFields,
    SelfCheckResult,
)

from day05_prompts import (
    EXTRACT_MEANING_SYSTEM,
    BUILD_FIELDS_SYSTEM,
    GENERATE_ANSWER_SYSTEM,
    SELF_CHECK_SYSTEM,
)

from day06_guardrails import safe_llm_call

from day06_prompts import (
    JSON_FALLBACK_SYSTEM,
    TEXT_FALLBACK_SYSTEM,
    SAFE_FALLBACK_RESPONSE,
)


load_dotenv()

client = LLMClient()
logger = logging.getLogger(__name__)


# SAFE STRUCTURED LLM REQUEST
def safe_json_request(system_prompt, user_prompt, schema):

    schema_json = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    return safe_llm_call(
        primary_func=lambda: client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0,
            json_mode=True,
        ),
        fallback_func=lambda: client.chat(
            system_prompt=JSON_FALLBACK_SYSTEM,
            user_prompt=(
                f"Задача:\n{system_prompt}\n\n"
                f"Входные данные:\n{user_prompt}\n\n"
                f"JSON-схема:\n{schema_json}"
            ),
            temperature=0,
            json_mode=True,
        ),
        schema=schema,
        max_length=4000,
        max_attempts=3,
    )


# SAFE TEXT GENERATION
def safe_text_request(system_prompt, user_prompt):

    return safe_llm_call(
        primary_func=lambda: client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            json_mode=False,
        ),
        fallback_func=lambda: client.chat(
            system_prompt=TEXT_FALLBACK_SYSTEM,
            user_prompt=user_prompt,
            temperature=0,
            json_mode=False,
        ),
        max_length=1500,
        max_attempts=3,
    )


def serialize(value):

    if value is None:
        return None

    return value.model_dump(mode="json")


def build_context(text, **stages):

    data = {
        "original_text": text,
        **stages,
    }

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


# FINAL PIPELINE
def run_pipeline(text: str) -> dict:

    logger.info("Pipeline started")

    result = {
        "input": text,
        "meaning": None,
        "classification": None,
        "structured_fields": None,
        "answer": None,
        "self_check": None,
        "status": "success",
        "failed_stages": [],
    }

    if not isinstance(text, str) or not text.strip():

        result["status"] = "invalid_input"
        result["answer"] = "Пожалуйста, введите обращение."

        return result

    # STEP 1 — EXTRACT MEANING
    logger.info("STEP 1: Extracting meaning")

    meaning = safe_json_request(
        EXTRACT_MEANING_SYSTEM,
        text,
        MeaningResult,
    )

    result["meaning"] = serialize(meaning)

    if meaning is None:
        result["failed_stages"].append("meaning")


    # STEP 2 — INTENT CLASSIFICATION
    logger.info("STEP 2: Classifying request")

    classification = safe_json_request(
        CLASSIFIER_SYSTEM,
        build_context(
            text,
            meaning=result["meaning"],
        ),
        IntentClassification,
    )

    result["classification"] = serialize(classification)

    if classification is None:
        result["failed_stages"].append("classification")


    # STEP 3 — STRUCTURED FIELDS
    logger.info("STEP 3: Building structured fields")

    fields = safe_json_request(
        BUILD_FIELDS_SYSTEM,
        build_context(
            text,
            meaning=result["meaning"],
            classification=result["classification"],
        ),
        StructuredFields,
    )

    result["structured_fields"] = serialize(fields)

    if fields is None:
        result["failed_stages"].append("structured_fields")


    # STEP 4 — GENERATE ANSWER
    logger.info("STEP 4: Generating answer")

    if classification is not None:
        instruction = select_response_prompt(
            classification.category
        )
    else:
        instruction = (
            "Дай краткий и полезный ответ. "
            "Не придумывай факты."
        )

    answer = safe_text_request(
        GENERATE_ANSWER_SYSTEM,
        build_context(
            text,
            meaning=result["meaning"],
            classification=result["classification"],
            structured_fields=result["structured_fields"],
            instruction=instruction,
        ),
    )

    if answer is None:

        result["answer"] = SAFE_FALLBACK_RESPONSE
        result["failed_stages"].append("answer")
        result["status"] = "fallback"

        logger.warning("Using safe fallback response")

        return result

    result["answer"] = answer


    # STEP 5 — SELF-CHECK
    logger.info("STEP 5: Checking answer")

    check = safe_json_request(
        SELF_CHECK_SYSTEM,
        build_context(
            text,
            meaning=result["meaning"],
            classification=result["classification"],
            structured_fields=result["structured_fields"],
            answer=result["answer"],
        ),
        SelfCheckResult,
    )

    result["self_check"] = serialize(check)

    if check is None:
        result["failed_stages"].append("self_check")


    # FINAL STATUS
    if check is not None and not check.is_valid:

        result["status"] = "needs_review"

    elif result["failed_stages"]:

        result["status"] = "degraded"

    else:

        result["status"] = "success"

    logger.info(
        "Pipeline completed. Status: %s",
        result["status"],
    )

    return result


# OUTPUT
def print_result(result: dict):

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


# DEMO MODE
def run_demo():

    demo_path = (
        Path(__file__).resolve().parent
        / "demo"
        / "inputs.json"
    )

    with demo_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        examples = json.load(file)

    for number, text in enumerate(examples, start=1):

        logger.info(
            "DEMO CASE %s/%s",
            number,
            len(examples),
        )

        print_result(run_pipeline(text))


# CLI
def main():

    parser = argparse.ArgumentParser(
        description="AI Customer Support Assistant"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--text",
        type=str,
        help="Customer request",
    )

    group.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration examples",
    )

    args = parser.parse_args()

    if args.demo:

        run_demo()

    elif args.text is not None:

        print_result(run_pipeline(args.text))

    else:

        text = input("Введите ваше обращение: ")

        print_result(run_pipeline(text))


if __name__ == "__main__":
    main()
