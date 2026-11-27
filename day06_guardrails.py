
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)

TRANSIENT_API_ERRORS = (
    TimeoutError,
    ConnectionError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)

import logging
import time

from pydantic import BaseModel, ValidationError


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# CUSTOM EXCEPTION
class GuardrailError(Exception):
    pass


# STEP 1 — RESPONSE VALIDATION
def validate_response(
    response: str,
    max_length: int = 4000,
) -> str:

    if not isinstance(response, str):
        raise GuardrailError("Response must be a string")

    if not response.strip():
        raise GuardrailError("Empty response")

    if len(response) > max_length:
        raise GuardrailError("Response is too long")

    return response.strip()


# STEP 2 — JSON VALIDATION
def validate_json(
    response: str,
    schema: type[BaseModel],
    max_length: int = 4000,
) -> BaseModel:

    response = validate_response(response, max_length)

    return schema.model_validate_json(response)


# STEP 3 — RETRY WITH EXPONENTIAL BACKOFF
def retry_call(
    func,
    max_attempts: int = 3,
    delay: float = 1.0,
    retry_exceptions=TRANSIENT_API_ERRORS,
):

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):

        try:
            logger.info(
                f"API attempt {attempt}/{max_attempts}"
            )

            return func()

        except retry_exceptions as error:

            logger.warning(
                f"Attempt {attempt} failed: {error}"
            )

            if attempt == max_attempts:
                logger.error("All retry attempts failed")
                raise

            wait_time = delay * (2 ** (attempt - 1))

            logger.info(
                f"Retrying in {wait_time} seconds"
            )

            time.sleep(wait_time)


# STEP 4 — SAFE CALL WITH FALLBACK
def safe_llm_call(
    primary_func,
    fallback_func=None,
    schema=None,
    max_length: int = 4000,
    max_attempts: int = 3,
    delay: float = 1.0,
    retry_exceptions=TRANSIENT_API_ERRORS,
):

    def validate(raw_response):

        if schema is not None:
            return validate_json(
                raw_response,
                schema,
                max_length,
            )

        return validate_response(
            raw_response,
            max_length,
        )

    handled_errors = (
    GuardrailError,
    ValidationError,
    APIError,
    ) + retry_exceptions

    # PRIMARY REQUEST
    try:
        logger.info("Starting primary LLM request")

        response = retry_call(
            primary_func,
            max_attempts=max_attempts,
            delay=delay,
            retry_exceptions=retry_exceptions,
        )

        result = validate(response)

        logger.info("Primary response is valid")

        return result

    except handled_errors as error:

        logger.warning(
            f"Primary request failed: "
            f"{type(error).__name__}: {error}"
        )

    # FALLBACK REQUEST
    if fallback_func is not None:

        try:
            logger.info("Starting fallback request")

            response = retry_call(
                fallback_func,
                max_attempts=max_attempts,
                delay=delay,
                retry_exceptions=retry_exceptions,
            )

            result = validate(response)

            logger.info("Fallback response is valid")

            return result

        except handled_errors as error:

            logger.error(
                f"Fallback failed: "
                f"{type(error).__name__}: {error}"
            )

    # SAFE FALLBACK
    logger.error(
        "Unable to obtain a valid LLM response"
    )

    return None
