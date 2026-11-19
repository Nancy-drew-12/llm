"""LLM pipeline: summary, 3 ключевые мысли и короткий ответ для каждого текста.

Два режима:
    python main.py                      # пайплайн с лучшим промптом -> results/day02_results.json
    python main.py my_text.txt          # свои файлы
    python main.py --compare            # День 2: сравнить все варианты промпта
    python main.py --compare --repeats 3
    python main.py --compare --variants v2_constraints v4_combined
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient
from prompts import (
    BEST_VARIANT,
    KEY_POINTS_COUNT,
    LIMITS,
    MAX_KEY_POINT_WORDS,
    MAX_RESPONSE_CHARS,
    MAX_SUMMARY_CHARS,
    PROMPT_VARIANTS,
    build_prompts,
)

INPUT_DIR = Path("sample_inputs")
RESULTS_PATH = Path("results/day02_results.json")
COMPARE_PATH = Path("results/day02_comparison.json")
REPORT_PATH = Path("reports/day02_prompt_comparison.md")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")


# ---------- работа с ответом модели ----------

def parse_json(raw):
    """Достаёт JSON из ответа модели. Модели иногда оборачивают его в ```json ... ```."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("В ответе модели нет JSON")
    return json.loads(match.group(0))


def normalize(data):
    """Приводит ответ модели к единому виду."""
    key_points = data.get("key_points", [])
    if isinstance(key_points, str):
        key_points = [key_points]
    return {
        "text_type": data.get("text_type", ""),
        "summary": str(data.get("summary", "")).strip(),
        "key_points": [str(p).strip() for p in key_points],
        "response": str(data.get("response", "")).strip(),
    }


def check_constraints(result):
    """Проверяет ограничения из ТЗ: длина summary, ровно 3 key points, короткий ответ."""
    summary_len = len(result["summary"])
    response_len = len(result["response"])
    kp_max_words = max((len(p.split()) for p in result["key_points"]), default=0)

    checks = {
        "summary_ok": 0 < summary_len <= MAX_SUMMARY_CHARS,
        "key_points_ok": len(result["key_points"]) == KEY_POINTS_COUNT,
        "key_point_words_ok": kp_max_words <= MAX_KEY_POINT_WORDS,
        "response_ok": 0 < response_len <= MAX_RESPONSE_CHARS,
    }
    violations = [name for name, ok in checks.items() if not ok]
    metrics = {"summary_len": summary_len, "response_len": response_len, **checks, "all_ok": not violations}
    return metrics, violations


def analyze_text(client, text, variant=BEST_VARIANT):
    system_prompt, user_prompt = build_prompts(text, variant)
    raw = client.chat(system_prompt, user_prompt, temperature=0.2, json_mode=True)
    return normalize(parse_json(raw))


# ---------- ввод / вывод ----------

def load_texts(paths=None):
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


def print_result(name, result, violations):
    print("\n" + "=" * 70)
    print(f"📄 {name}" + (f"  [{result['text_type']}]" if result.get("text_type") else ""))
    print("-" * 70)
    print(f"Summary: {result['summary']}")
    print("Ключевые мысли:")
    for i, point in enumerate(result["key_points"], 1):
        print(f"  {i}. {point}")
    print(f"Ответ: {result['response']}")
    print("Ограничения: " + ("✅ соблюдены" if not violations else "⚠️ нарушены: " + ", ".join(violations)))


def save_json(path, payload):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- режим 1: пайплайн с лучшим промптом ----------

def run_pipeline(client, texts, variant):
    logger.info("Вариант промпта: %s", variant)
    results = []
    for name, text in texts:
        logger.info("Обрабатываю %s", name)
        try:
            result = analyze_text(client, text, variant)
            metrics, violations = check_constraints(result)
            status = "ok"
        except Exception as e:
            logger.error("Ошибка на %s: %s", name, e)
            result = {"text_type": "", "summary": "", "key_points": [], "response": "", "error": str(e)}
            metrics, violations, status = {}, ["error"], "error"

        print_result(name, result, violations)
        results.append({"file": name, "status": status, **result, "checks": metrics})

    save_json(RESULTS_PATH, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "prompt_variant": variant,
        "results": results,
    })
    logger.info("Результаты сохранены в %s", RESULTS_PATH)


# ---------- режим 2: сравнение вариантов промпта ----------

def run_once(client, text, variant):
    start = time.perf_counter()
    try:
        result = analyze_text(client, text, variant)
    except Exception as e:
        return {"json_ok": False, "error": str(e), "latency_s": round(time.perf_counter() - start, 2)}
    metrics, violations = check_constraints(result)
    return {"json_ok": True, **result, **metrics, "violations": violations,
            "latency_s": round(time.perf_counter() - start, 2)}


def summarize(runs):
    n = len(runs)
    ok = [r for r in runs if r["json_ok"]]

    def share(key):
        return round(sum(r.get(key, False) for r in ok) / n, 2) if n else 0

    def mean(key):
        return round(sum(r[key] for r in ok) / len(ok), 1) if ok else 0

    return {
        "runs": n,
        "json_ok": round(len(ok) / n, 2) if n else 0,
        "summary_ok": share("summary_ok"),
        "key_points_ok": share("key_points_ok"),
        "key_point_words_ok": share("key_point_words_ok"),
        "response_ok": share("response_ok"),
        "all_ok": share("all_ok"),
        "avg_summary_len": mean("summary_len"),
        "avg_response_len": mean("response_len"),
        "avg_latency_s": mean("latency_s"),
    }


def build_report(model, repeats, summary, runs_by_variant, texts):
    lines = [
        "# День 2: сравнение промптов",
        "",
        f"- Модель: `{model}`, temperature = 0.2 (одинаковая для всех вариантов)",
        f"- Тексты: {len(texts)} шт., прогонов на текст: {repeats}",
        f"- Лимиты: summary ≤ {LIMITS['max_summary_chars']} симв., key points = {LIMITS['key_points_count']} "
        f"(≤ {LIMITS['max_key_point_words']} слов), response ≤ {LIMITS['max_response_chars']} симв.",
        "",
        "## Метрики (доля прогонов, где условие выполнено)",
        "",
        "| Вариант | JSON ок | summary ок | 3 key points | key points ≤ слов | response ок | **всё ок** "
        "| ср. длина summary | ср. длина response | ср. время, с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, s in summary.items():
        lines.append(
            f"| {variant} | {s['json_ok']:.0%} | {s['summary_ok']:.0%} | {s['key_points_ok']:.0%} | "
            f"{s['key_point_words_ok']:.0%} | {s['response_ok']:.0%} | **{s['all_ok']:.0%}** | "
            f"{s['avg_summary_len']} | {s['avg_response_len']} | {s['avg_latency_s']} |"
        )

    lines += ["", "## Ответы по текстам (первый прогон)", ""]
    for name, _ in texts:
        lines += [f"### {name}", ""]
        for variant, runs in runs_by_variant.items():
            run = next(r for r in runs if r["file"] == name)
            lines.append(f"**{variant}**" + (f" — `{run.get('text_type')}`" if run.get("text_type") else ""))
            if not run["json_ok"]:
                lines += [f"- ❌ ошибка: {run['error']}", ""]
                continue
            lines.append(f"- summary ({run['summary_len']} симв.): {run['summary']}")
            lines += [f"  - {p}" for p in run["key_points"]]
            lines.append(f"- response ({run['response_len']} симв.): {run['response']}")
            if run["violations"]:
                lines.append(f"- ⚠️ нарушения: {', '.join(run['violations'])}")
            lines.append("")
    return "\n".join(lines)


def run_comparison(client, texts, repeats, variants):
    runs_by_variant = {}
    for variant in variants:
        runs = []
        for name, text in texts:
            for i in range(repeats):
                logger.info("%s | %s | прогон %d", variant, name, i + 1)
                runs.append({"file": name, "repeat": i + 1, **run_once(client, text, variant)})
        runs_by_variant[variant] = runs

    summary = {v: summarize(runs) for v, runs in runs_by_variant.items()}
    save_json(COMPARE_PATH, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "repeats": repeats,
        "summary": summary,
        "runs": runs_by_variant,
    })
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build_report(client.model, repeats, summary, runs_by_variant, texts), encoding="utf-8")

    print("\nИтог (доля прогонов, где соблюдены ВСЕ ограничения):")
    for variant, s in summary.items():
        print(f"  {variant:16s} all_ok={s['all_ok']:.0%}  json_ok={s['json_ok']:.0%}  "
              f"summary≈{s['avg_summary_len']}  response≈{s['avg_response_len']}  {s['avg_latency_s']} с")
    logger.info("Отчёт: %s", REPORT_PATH)


def main():
    parser = argparse.ArgumentParser(description="LLM pipeline")
    parser.add_argument("files", nargs="*", help="Файлы с текстами (по умолчанию sample_inputs/*.txt)")
    parser.add_argument("--compare", action="store_true", help="Сравнить все варианты промпта")
    parser.add_argument("--repeats", type=int, default=2, help="Прогонов на текст в режиме --compare")
    parser.add_argument("--variant", default=BEST_VARIANT, choices=list(PROMPT_VARIANTS), help="Вариант промпта")
    parser.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS), choices=list(PROMPT_VARIANTS),
                        help="Какие варианты сравнивать в режиме --compare (по умолчанию все)")
    args = parser.parse_args()

    load_dotenv()
    client = LLMClient()

    texts = load_texts(args.files)
    if not texts:
        logger.error("Нет текстов: положи .txt в %s или передай пути аргументами", INPUT_DIR)
        sys.exit(1)

    if args.compare:
        run_comparison(client, texts, args.repeats, args.variants)
    else:
        run_pipeline(client, texts, args.variant)


if __name__ == "__main__":
    main()