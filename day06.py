
import logging
import json

from dotenv import load_dotenv

from llm_client import LLMClient
from day05_schemas import MeaningResult
from day05_prompts import EXTRACT_MEANING_SYSTEM

from day06_guardrails import safe_llm_call

from day06_prompts import (
    JSON_FALLBACK_SYSTEM,
    TEXT_FALLBACK_SYSTEM,
    SAFE_FALLBACK_RESPONSE,
)


load_dotenv()

client = LLMClient()
logger = logging.getLogger(__name__)


# STEP 1 — SAFE MEANING EXTRACTION
def extract_meaning(text: str) -> MeaningResult | None:

    logger.info("STEP 1: Extracting meaning")

    schema_json = json.dumps(
        MeaningResult.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    return safe_llm_call(
        primary_func=lambda: client.chat(
            system_prompt=EXTRACT_MEANING_SYSTEM,
            user_prompt=text,
            temperature=0,
            json_mode=True,
        ),

        fallback_func=lambda: client.chat(
            system_prompt=JSON_FALLBACK_SYSTEM,
            user_prompt=(
                f"Исходное сообщение:\n{text}\n\n"
                f"JSON-схема:\n{schema_json}"
            ),
            temperature=0,
            json_mode=True,
        ),

        schema=MeaningResult,
        max_length=4000,
        max_attempts=3,
    )


# STEP 2 — SAFE ANSWER GENERATION
def generate_answer(
    text: str,
    meaning: MeaningResult | None,
) -> str:

    logger.info("STEP 2: Generating answer")

    context = (
        meaning.model_dump_json(indent=2)
        if meaning is not None
        else "Структурированный анализ недоступен"
    )

    user_prompt = (
        f"Исходное обращение:\n{text}\n\n"
        f"Анализ обращения:\n{context}"
    )

    result = safe_llm_call(
        primary_func=lambda: client.chat(
            system_prompt=(
                "Ты ассистент службы поддержки. "
                "Дай полезный и краткий ответ. "
                "Не придумывай факты. "
                "Не обещай действий от имени компании. "
                "Отвечай на русском языке."
            ),
            user_prompt=user_prompt,
            temperature=0.3,
            json_mode=False,
        ),

        fallback_func=lambda: client.chat(
            system_prompt=TEXT_FALLBACK_SYSTEM,
            user_prompt=text,
            temperature=0,
            json_mode=False,
        ),

        max_length=1500,
        max_attempts=3,
    )

    if result is None:
        logger.warning("Using predefined safe fallback")
        return SAFE_FALLBACK_RESPONSE

    return result


# STEP 3 — RUN PIPELINE
def run_pipeline(text: str) -> str:

    logger.info("Pipeline started")

    if not isinstance(text, str) or not text.strip():
        logger.warning("Invalid user input")
        return "Пожалуйста, введите ваше обращение."

    meaning = extract_meaning(text)

    if meaning is None:
        logger.warning(
            "Meaning extraction failed. "
            "Continuing in degraded mode."
        )
    else:
        logger.info("Meaning extraction completed")

        print("\n[EXTRACTED MEANING]")
        print(meaning.model_dump_json(indent=2))

    answer = generate_answer(text, meaning)

    logger.info("Pipeline completed")

    return answer


# MAIN
if __name__ == "__main__":

    text = (
        "Уже неделю не могу войти в аккаунт, "
        "а поддержка молчит!"
    )

    print("\n[INPUT]")
    print(text)

    answer = run_pipeline(text)

    print("\n[FINAL ANSWER]")
    print(answer)
