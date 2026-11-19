"""LLM pipeline: анализ входящего текста и ответ на него.

Режимы:
    python main.py                      # День 3: structured output + валидация схемой -> results/day03_results.json
    python main.py my_text.txt          # свои файлы
    python main.py --demo-errors        # День 3: показать, как разбираются сломанные ответы (без LLM)
    python main.py --day2               # пайплайн Дня 2 (лучший промпт v4, без схемы)
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
    build_structured_prompts,
)
from schemas import LLMOutputError, TextAnalysis, parse_analysis

INPUT_DIR = Path("sample_inputs")
RESULTS_PATH = Path("results/day02_results.json")
DAY3_RESULTS_PATH = Path("results/day03_results.json")
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


# ---------- День 3: structured output ----------

# Куда отправляем обращение в зависимости от категории
ROUTES = {
    "complaint": "support",        # в очередь поддержки, ответ клиенту — черновик для оператора
    "question": "auto_reply",      # можно отвечать автоматически
    "request": "assistant",        # отдаём пользователю готовый результат (письмо, текст)
    "news": "digest",              # в дайджест новостей, клиенту не отвечаем
    "other": "manual_review",
}


def analyze_structured(client, text):
    """Вызов модели + разбор ответа по схеме. Возвращает TextAnalysis или бросает LLMOutputError."""
    system_prompt, user_prompt = build_structured_prompts(text)
    raw = client.chat(system_prompt, user_prompt, temperature=0.2, json_mode=True)
    return parse_analysis(raw)


def decide(analysis: TextAnalysis):
    """Используем поля structured output для решений, а не только для печати."""
    category = analysis.category.value
    sentiment = analysis.sentiment.value

    if category == "complaint" and sentiment == "negative":
        priority = "high"
    elif category in ("complaint", "request") or sentiment == "negative":
        priority = "medium"
    else:
        priority = "low"

    route = ROUTES[category]
    # Позитивный отзыв не требует ручного разбора — отвечаем благодарностью
    if category == "other" and sentiment == "positive":
        route = "auto_reply"

    # Как поступаем с final_answer:
    # auto  — отправляем автору сразу; draft — черновик оператору (модель может пообещать лишнее);
    # none  — автору не отвечаем (новости, ручной разбор)
    if route in ("auto_reply", "assistant"):
        reply_mode = "auto"
    elif route == "support":
        reply_mode = "draft"
    else:
        reply_mode = "none"

    return {"priority": priority, "route": route, "reply_mode": reply_mode, "final_answer": analysis.final_answer}

def length_warnings(analysis: TextAnalysis):
    """Мягкие проверки длины из Дня 2: не ломают пайплайн, а только предупреждают."""
    warnings = []
    if len(analysis.summary) > MAX_SUMMARY_CHARS:
        warnings.append(f"summary длиннее {MAX_SUMMARY_CHARS} символов ({len(analysis.summary)})")
    if len(analysis.final_answer) > MAX_RESPONSE_CHARS:
        warnings.append(f"final_answer длиннее {MAX_RESPONSE_CHARS} символов ({len(analysis.final_answer)})")
    long_points = [p for p in analysis.key_points if len(p.split()) > MAX_KEY_POINT_WORDS]
    if long_points:
        warnings.append(f"{len(long_points)} key point(s) длиннее {MAX_KEY_POINT_WORDS} слов")
    return warnings


def print_structured(name, analysis, decision, warnings):
    print("\n" + "=" * 70)
    print(f"📄 {name}  [{analysis.category.value} · {analysis.sentiment.value}]  "
          f"приоритет: {decision['priority']}  →  {decision['route']}")
    print("-" * 70)
    print(f"Summary: {analysis.summary}")
    print("Ключевые мысли:")
    for i, point in enumerate(analysis.key_points, 1):
        print(f"  {i}. {point}")
    if decision["reply_mode"] == "auto":
        print(f"Ответ автору (отправлен): {decision['final_answer']}")
    elif decision["reply_mode"] == "draft":
        print(f"Черновик для оператора: {decision['final_answer']}")
    else:
        print(f"Автору не отвечаем (маршрут {decision['route']}). Вывод: {decision['final_answer']}")
    if warnings:
        print("⚠️ " + "; ".join(warnings))


def print_failure(name, error):
    print("\n" + "=" * 70)
    print(f"📄 {name}  ❌ ответ модели не принят [{error.kind}]")
    print("-" * 70)
    print(error.message)
    print("→ обращение отправлено на ручной разбор (manual_review)")


def run_structured(client, texts):
    results = []
    for name, text in texts:
        logger.info("Обрабатываю %s", name)
        try:
            analysis = analyze_structured(client, text)
        except LLMOutputError as e:
            logger.error("%s: %s", name, e.message)
            print_failure(name, e)
            results.append({"file": name, "status": "error", "error_kind": e.kind,
                            "error_message": e.message, "raw": e.raw[:500],
                            "decision": {"priority": "medium", "route": "manual_review", "reply_mode": "none"}})
            continue
        except Exception as e:  # сеть, таймаут, Ollama не запущена
            logger.error("%s: ошибка API: %s", name, e)
            err = LLMOutputError("api_error", f"Не удалось получить ответ модели: {e}")
            print_failure(name, err)
            results.append({"file": name, "status": "error", "error_kind": "api_error",
                            "error_message": err.message,
                            "decision": {"priority": "medium", "route": "manual_review", "reply_mode": "none"}})
            continue

        decision = decide(analysis)
        warnings = length_warnings(analysis)
        print_structured(name, analysis, decision, warnings)
        results.append({"file": name, "status": "ok", "analysis": analysis.model_dump(mode="json"),
                        "decision": decision, "warnings": warnings})

    stats = build_stats(results)
    print_stats(stats)
    save_json(DAY3_RESULTS_PATH, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "stats": stats,
        "results": results,
    })
    logger.info("Результаты сохранены в %s", DAY3_RESULTS_PATH)


def build_stats(results):
    """Сводка по батчу — ещё одно использование structured полей."""
    ok = [r for r in results if r["status"] == "ok"]

    def count(values):
        out = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return out

    return {
        "total": len(results),
        "ok": len(ok),
        "errors": count(r["error_kind"] for r in results if r["status"] == "error"),
        "by_category": count(r["analysis"]["category"] for r in ok),
        "by_sentiment": count(r["analysis"]["sentiment"] for r in ok),
        "by_route": count(r["decision"]["route"] for r in results),
        "high_priority": [r["file"] for r in ok if r["decision"]["priority"] == "high"],
    }


def print_stats(stats):
    print("\n" + "=" * 70)
    print(f"Итого: {stats['ok']}/{stats['total']} ответов прошли валидацию схемой")
    if stats["errors"]:
        print(f"Ошибки: {stats['errors']}")
    print(f"Категории: {stats['by_category']}")
    print(f"Тональность: {stats['by_sentiment']}")
    print(f"Маршруты: {stats['by_route']}")
    if stats["high_priority"]:
        print(f"🔥 Высокий приоритет: {', '.join(stats['high_priority'])}")


# Заранее сломанные ответы модели — чтобы проверить обработку ошибок без LLM
BROKEN_EXAMPLES = {
    "текст вместо JSON": "Конечно! Вот анализ: категория — жалоба, тон негативный.",
    "битый JSON (лишняя запятая)": '{"category": "complaint", "sentiment": "negative", "summary": "S", "key_points": ["a", "b", "c"], "final_answer": "Извините",}',
    "нет полей": '{"category": "complaint", "summary": "S", "key_points": ["a", "b", "c"]}',
    "неверные типы": '{"category": "complaint", "sentiment": "negative", "summary": "S", "key_points": "a, b, c", "final_answer": 42}',
    "недопустимые значения": '{"category": "жалоба", "sentiment": "mixed", "summary": "S", "key_points": ["a", "b", "c", "d"], "final_answer": "Ок"}',
    "пустой ответ": "",
    "валидный, но в markdown": '```json\n{"category": "Question ", "sentiment": "Neutral", "summary": "S", "key_points": ["a", "b", "c"], "final_answer": "Да"}\n```',
}


def demo_errors():
    """Прогоняет сломанные ответы через parse_analysis и печатает, что программа о них скажет."""
    for name, raw in BROKEN_EXAMPLES.items():
        try:
            analysis = parse_analysis(raw)
            print(f"✅ {name}: принят — {analysis.category.value} / {analysis.sentiment.value}")
        except LLMOutputError as e:
            print(f"❌ {name} [{e.kind}]: {e.message}")

def main():
    parser = argparse.ArgumentParser(description="LLM pipeline")
    parser.add_argument("files", nargs="*", help="Файлы с текстами (по умолчанию sample_inputs/*.txt)")
    parser.add_argument("--day2", action="store_true", help="Пайплайн Дня 2 (без схемы)")
    parser.add_argument("--compare", action="store_true", help="Сравнить все варианты промпта (День 2)")
    parser.add_argument("--repeats", type=int, default=2, help="Прогонов на текст в режиме --compare")
    parser.add_argument("--variant", default=BEST_VARIANT, choices=list(PROMPT_VARIANTS), help="Вариант промпта для --day2")
    parser.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS), choices=list(PROMPT_VARIANTS),
                        help="Какие варианты сравнивать в режиме --compare (по умолчанию все)")
    parser.add_argument("--demo-errors", action="store_true", help="Показать обработку сломанных ответов (без LLM)")
    args = parser.parse_args()

    if args.demo_errors:
        demo_errors()
        return

    load_dotenv()
    client = LLMClient()

    texts = load_texts(args.files)
    if not texts:
        logger.error("Нет текстов: положи .txt в %s или передай пути аргументами", INPUT_DIR)
        sys.exit(1)

    if args.compare:
        run_comparison(client, texts, args.repeats, args.variants)
    elif args.day2:
        run_pipeline(client, texts, args.variant)
    else:
        run_structured(client, texts)


if __name__ == "__main__":
    main()