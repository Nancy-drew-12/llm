from dotenv import load_dotenv

from llm_client import LLMClient
from day04_schemas import IntentClassification
from day04_prompts import CLASSIFIER_SYSTEM
from router import select_response_prompt


load_dotenv()
client = LLMClient()


def classify(text: str) -> IntentClassification:
    raw = client.chat(
        system_prompt=CLASSIFIER_SYSTEM,
        user_prompt=text,
        temperature=0,
        json_mode=True,
    )

    return IntentClassification.model_validate_json(raw)


def process(text: str) -> dict:
    # 1. Определяем категорию и intent
    classification = classify(text)

    # 2. Python выбирает инструкцию
    instruction = select_response_prompt(
        classification.category
    )
    print("\n[ROUTING DEBUG]")
    print(f"Category: {classification.category.value}")
    print(f"Intent: {classification.intent}")
    print(f"Selected instruction: {instruction}")

    # 3. Генерируем ответ
    response = client.chat(
        system_prompt=instruction,
        user_prompt=text,
        temperature=0.3,
    )

    return {
        "category": classification.category.value,
        "intent": classification.intent,
        "response": response,
    }

from test_cases import TEST_CASES


def run_tests():
    correct = 0

    for text, expected in TEST_CASES:
        result = process(text)

        predicted = result["category"]

        if predicted == expected:
            correct += 1

        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Response: {result['response']}")
        print("-" * 40)

    print(f"Accuracy: {correct}/{len(TEST_CASES)}")

if __name__ == "__main__":
    run_tests()