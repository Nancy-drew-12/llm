# AI Customer Support Assistant

An LLM-powered customer support assistant that analyzes customer requests, classifies intents, extracts structured information, generates responses, and performs automated quality checks.

The project demonstrates practical LLM engineering techniques, including prompt engineering, structured outputs, routing, guardrails, retry mechanisms, fallback strategies, and automated testing.

## Features

- Customer request analysis
- Intent classification
- Structured JSON extraction with Pydantic
- Intent-based prompt routing
- Answer generation
- LLM-based self-check
- Retry with exponential backoff
- Fallback prompts and predefined safe responses
- Graceful degradation
- Logging and error handling
- Command-line interface (CLI)
- Seven demonstration scenarios
- Fourteen automated tests

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Qwen2.5:7b | Default language model |
| Ollama | Local LLM inference |
| OpenAI SDK | OpenAI-compatible API client |
| Pydantic | Structured output validation |
| python-dotenv | Environment configuration |
| argparse | Command-line interface |
| unittest | Automated testing |
| Mock | Failure simulation |
| Logging | Pipeline monitoring |

## Pipeline Architecture

```text
User Input
    |
    v
Meaning Extraction
    |
    v
Intent Classification
    |
    v
Structured Field Extraction
    |
    v
Intent-based Answer Generation
    |
    v
Self-check
    |
    v
Final JSON Output
```

Each LLM stage is protected by validation, retry and fallback mechanisms.

If an intermediate stage fails, the pipeline attempts to continue using the available information.

## Project Structure

```text
llm/
├── llm_client.py
├── router.py
├── day04.py
├── day04_prompts.py
├── day04_schemas.py
├── day05.py
├── day05_prompts.py
├── day05_schemas.py
├── day06.py
├── day06_guardrails.py
├── day06_prompts.py
├── day06_tests.py
├── day07.py
├── day07_tests.py
├── demo/
│   └── inputs.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Earlier development stages are preserved for educational purposes.

The final application entry point is `day07.py`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nancy-drew-12/llm.git
cd llm
```

### 2. Create a virtual environment

The project was developed and tested with Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

The application supports OpenAI-compatible API providers.

#### Option A: Local Ollama (default)

Install Ollama from https://ollama.com.

Download the model:

```bash
ollama pull qwen2.5:7b
```

Make sure the Ollama server is running. If necessary, start it in a separate terminal:

```bash
ollama serve
```

Configure `.env`:

```dotenv
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
```

A paid API key is not required for local Ollama inference.

#### Option B: OpenAI

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

#### Option C: OpenRouter

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your_model_name
```

Select a model that supports the required chat completion and JSON output functionality.

Never commit real API keys to GitHub.

## Usage

### Interactive mode

```bash
python day07.py
```

### Single request

```bash
python day07.py --text "Уже неделю не могу войти в аккаунт"
```

### Demo mode

```bash
python day07.py --demo
```

The demo mode processes seven predefined customer requests from `demo/inputs.json`.

## Intent Categories

| Category | Description |
|---|---|
| support | Technical or account-related issues |
| complaint | Customer dissatisfaction |
| feedback | Suggestions and feature requests |
| sales | Pricing and subscription questions |
| general_question | General information requests |

## Example Input

```text
Уже неделю не могу войти в аккаунт
```

## Example Output

Illustrative JSON output. Actual model responses may vary.

```json
{
  "input": "Уже неделю не могу войти в аккаунт",
  "meaning": {
    "summary": "Пользователь не может войти в аккаунт уже неделю.",
    "user_goal": "Восстановить доступ к аккаунту.",
    "key_details": [
      "Проблема со входом",
      "Уже неделю"
    ]
  },
  "classification": {
    "category": "support",
    "intent": "resolve_login_issue"
  },
  "structured_fields": {
    "problem": "Не удаётся войти в аккаунт",
    "duration": "уже неделю",
    "requested_action": "Восстановить доступ",
    "important_facts": [
      "Проблема сохраняется неделю"
    ]
  },
  "answer": "Уточните, пожалуйста, какая ошибка появляется при попытке входа в аккаунт.",
  "self_check": {
    "is_valid": true,
    "contradictions": [],
    "missing_details": [],
    "unsupported_claims": [],
    "explanation": "Ответ соответствует запросу пользователя."
  },
  "status": "success",
  "failed_stages": []
}
```

## Pipeline Statuses

| Status | Description |
|---|---|
| success | All stages completed and self-check approved the answer |
| needs_review | Self-check detected potential quality issues |
| degraded | One or more stages failed, but an answer was generated |
| fallback | A predefined safe response was returned |
| invalid_input | The input was invalid or empty |

## Reliability and Guardrails

The application handles:

- Empty LLM responses
- Invalid JSON
- Missing required fields
- Excessively long responses
- Temporary API errors
- API retry exhaustion
- Intermediate pipeline failures

Reliability mechanisms include:

- Pydantic schema validation
- Response length validation
- Exponential backoff
- Up to three application-level attempts per request
- Alternative fallback prompts
- Predefined safe responses
- Graceful degradation
- Stage-level logging

## Automated Testing

Run guardrail tests:

```bash
python day06_tests.py
```

Run final pipeline tests:

```bash
python day07_tests.py
```

Run all tests:

```bash
python -m unittest day06_tests day07_tests -v
```

Tested scenarios include invalid JSON, missing fields, temporary API failures, empty responses, excessive response length, partial pipeline failures and fallback behavior.

Mock-based tests do not require actual LLM inference.

## Demo Scenarios

| Scenario | Expected category |
|---|---|
| Account login issue | support |
| Subscription activation issue | support |
| Application freezes | complaint |
| Feature request | feedback |
| Pricing and annual subscription | sales |
| Changing email address | general_question |
| Missing message history | support |

## Evaluation Results

Results from the recorded local demonstration run:

| Metric | Result |
|---|---|
| Processed demo scenarios | 7/7 |
| Completed pipeline stages | 35/35 |
| Successful API responses | 35/35 |
| Self-check approved | 3/7 |
| Requires review | 4/7 |
| Automated tests passed | 14/14 |

Technical pipeline execution and generated response quality are evaluated separately.

## Known Limitations

- LLM-generated responses may contain hallucinations.
- Self-check can produce false positives and false negatives.
- Schema validation guarantees structural conformity, not factual accuracy.
- The application does not have access to actual customer accounts, company policies or product documentation.
- Generated troubleshooting recommendations may be inappropriate or unsafe.
- Answers marked as `needs_review` require human review.
- Even answers marked as `success` are not guaranteed to be factually correct.
- Response generation may be slow when running a local model.
- Compatibility with alternative API providers has not been fully tested.

This is an educational prototype, not a production-ready autonomous customer support system.

## Future Improvements

- Retrieval-Augmented Generation (RAG)
- Company-specific knowledge base
- Automatic response correction
- Stronger safety and factuality checks
- Evaluation datasets and quality metrics
- FastAPI endpoint
- Web interface

## Project Summary

This project demonstrates the development of an LLM application from basic API integration to a structured, fault-tolerant mini-product.

It combines prompt engineering, intent classification, structured generation, validation, routing, error handling and automated testing in a single application.