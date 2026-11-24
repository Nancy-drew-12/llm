from day04_schemas import RequestCategory
from day04_prompts import RESPONSE_INSTRUCTIONS


def select_response_prompt(category: RequestCategory) -> str:
    return RESPONSE_INSTRUCTIONS[category.value]