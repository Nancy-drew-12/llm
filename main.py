"""День 1: базовый LLM pipeline.

Берёт все .txt из sample_inputs/ (или файлы из аргументов), для каждого получает
summary, 3 ключевые мысли и короткий ответ, печатает их и сохраняет в results/.

Запуск:
    python main.py
    python main.py path/to/text1.txt path/to/text2.txt
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient
from prompts import SYSTEM_PROMPT, build_analyze_prompt

INPUT_DIR = Path("sample_inputs")
OUTPUT_PATH = Path("results/day01_results.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")


def parse_json(raw):
    """Достаёт JSON из ответа модели. Модели иногда оборачивают его в ```json ... ```."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("В ответе модели нет JSON")
    return json.loads(match.group(0))


def analyze_text(client, text):
    raw = client.chat(SYSTEM_PROMPT, build_analyze_prompt(text), temperature=0.2, json_mode=True)
    data = parse_json(raw)
    return {
        "summary": data.get("summary", ""),
        "key_points": list(data.get("key_points", []))[:3],
        "response": data.get("response", ""),
    }


def load_texts(paths):
    if not paths:
        paths = sorted(INPUT_DIR.glob("*.txt"))
    texts = []
    for path in map(Path, paths):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append((path.name, text))
        else:
            logger.warning("Файл %s пустой, пропускаю", path)
    return texts


def print_result(name, result):
    print("\n" + "=" * 70)
    print(f"📄 {name}")
    print("-" * 70)
    print(f"Summary: {result['summary']}")
    print("Ключевые мысли:")
    for i, point in enumerate(result["key_points"], 1):
        print(f"  {i}. {point}")
    print(f"Ответ: {result['response']}")


def main():
    load_dotenv()
    client = LLMClient()

    texts = load_texts(sys.argv[1:])
    if not texts:
        logger.error("Нет текстов: положи .txt в %s или передай пути аргументами", INPUT_DIR)
        sys.exit(1)

    results = []
    for name, text in texts:
        logger.info("Обрабатываю %s", name)
        try:
            result = analyze_text(client, text)
            status = "ok"
        except Exception as e:
            logger.error("Ошибка на %s: %s", name, e)
            result = {"summary": "", "key_points": [], "response": "", "error": str(e)}
            status = "error"

        print_result(name, result)
        results.append({"file": name, "status": status, "model": client.model, **result})

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "results": results}
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Результаты сохранены в %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
