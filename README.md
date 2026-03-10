# Arabic HELM Evaluation

Evaluate Arabic language models on standardized benchmarks using Stanford's [HELM](https://github.com/stanford-crfm/helm) (Holistic Evaluation of Language Models) framework. This project provides a thin configuration and automation layer on top of HELM, supporting local models via [LM Studio](https://lmstudio.ai/) and cloud providers like [Fireworks AI](https://fireworks.ai/), with automatic result storage in PostgreSQL.

Built for the [HELM Arabic Leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html).

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Available Benchmarks](#available-benchmarks)
- [Currently Configured Models](#currently-configured-models)
- [Adding a New Model](#adding-a-new-model)
- [Configuration Files](#configuration-files)
- [Database Storage](#database-storage)
- [Disabling Thinking Mode](#disabling-thinking-mode)
- [Custom Client: FireworksNoThinkingClient](#custom-client-fireworksnothinkingclient)
- [Advanced: Raw HELM Commands](#advanced-raw-helm-commands)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Architecture](#architecture)

---

## Features

- **One-command evaluation pipeline** — configure, run, and store results with a single `helm_eval.py` invocation
- **Auto-generated YAML configs** — model deployments, tokenizers, and metadata are created/updated automatically from CLI flags
- **Thinking mode disabled** — custom client ensures fair comparison on the Arabic leaderboard by suppressing chain-of-thought reasoning
- **Arabic MCQ system prompt** — injects an Arabic instruction to respond with answer letters only, preventing regex extraction failures
- **PostgreSQL streaming** — results are streamed to the database in batches of 10, with git commit/branch tracked per run
- **Parallel evaluation** — pass `-n <threads>` to run evaluation with multiple threads
- **Multiple providers** — supports any OpenAI-compatible API (LM Studio, Fireworks AI, etc.)
- **Temp file cleanup** — generated run spec files are automatically deleted after evaluation

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10** | Required by HELM |
| **HELM** | Installed via `crfm-helm` pip package in a virtual environment |
| **API access** | Local model via [LM Studio](https://lmstudio.ai/) **or** a [Fireworks AI](https://fireworks.ai/) API key |
| **PostgreSQL** | Optional — only needed for storing evaluation results |
| **HuggingFace tokenizer** | The tokenizer model must be accessible (downloads automatically) |

### Python dependencies

The following packages are required beyond what HELM provides:

| Package | Purpose |
|---------|---------|
| `psycopg2` | PostgreSQL database connection |
| `python-dotenv` | Load `.env` database credentials |
| `ijson` | Stream large JSON files without loading into memory |
| `pyyaml` | Read/write YAML configuration files |

---

## Installation

### 1. Create and activate the virtual environment

```bash
python3.10 -m venv helm-env
source helm-env/bin/activate
```

### 2. Install HELM

```bash
pip install crfm-helm
```

### 3. Install additional dependencies

```bash
pip install psycopg2-binary python-dotenv ijson pyyaml
```

### 4. Clone this repository

```bash
git clone https://github.com/ahmed-inceptlabs/arabic-helm-eval.git
cd arabic-helm-eval
```

### 5. Configure credentials

Create `credentials.conf` in the project root:

```
openaiApiKey: "lm-studio"
fireworksApiKey: "your-fireworks-api-key"
```

HELM resolves API keys by convention: a deployment named `fireworks/model-name` looks for `fireworksApiKey`, and `local/model-name` or OpenAI-compatible endpoints look for `openaiApiKey`.

---

## Quick Start

### Evaluate a Fireworks AI model

```bash
source helm-env/bin/activate

python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --tokenizer Qwen/Qwen2.5-7B \
  --benchmark aratrust \
  --suite my-experiment \
  --max-instances 600
```

### Evaluate a local LM Studio model

```bash
python helm_eval.py \
  --model-name local/qwen3-5-9b \
  --api-base http://127.0.0.1:1234/v1 \
  --api-key lm-studio \
  --tokenizer Qwen/Qwen3.5-9B \
  --benchmark arabic_mmlu \
  --suite local-test \
  --max-instances 10
```

### Run all benchmarks with parallel threads

```bash
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --suite base-kimi-k2p5 \
  --max-instances 600 \
  -n 8
```

When `--tokenizer` and `--benchmark` are omitted, they default to `Qwen/Qwen2.5-7B` and `aratrust` respectively. The `-n 8` flag runs evaluation with 8 parallel threads for faster execution.

### What happens under the hood

1. **Configure** — updates `model_deployments.yaml`, `tokenizer_configs.yaml`, `model_metadata.yaml`, and `credentials.conf` with the provided arguments
2. **Generate run spec** — creates a temporary `run_specs_<suite>.conf` file
3. **Run HELM** — executes `helm-run` with `PYTHONPATH=.` so the custom client is discoverable
4. **Store results** — streams evaluation results to PostgreSQL in batches of 10
5. **Clean up** — removes the generated run spec file

---

## CLI Reference

### `helm_eval.py` — Unified evaluation pipeline

```
python helm_eval.py [OPTIONS]
```

#### Required arguments

| Flag | Description | Example |
|------|-------------|---------|
| `--model-name` | Model identifier (`provider/name` format) | `fireworks/kimi-k2p5` |
| `--api-base` | API base URL | `https://api.fireworks.ai/inference/v1` |
| `--suite` | Suite name (used for output directory and DB tracking) | `my-experiment` |

#### Optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--api-model` | `None` | OpenAI model name for the provider. When set, uses `FireworksNoThinkingClient`; when omitted, uses HELM's standard `OpenAIClient` |
| `--api-key` | `None` | API key. If not set, uses existing value in `credentials.conf` |
| `--tokenizer` | `Qwen/Qwen2.5-7B` | HuggingFace tokenizer identifier |
| `--max-seq-len` | `131072` | Maximum sequence length |
| `--benchmark` | `aratrust` | Benchmark to run (see [Available Benchmarks](#available-benchmarks)) |
| `--benchmark-args` | Benchmark-specific default | Override benchmark arguments (e.g. `category=ethics`) |
| `--max-instances` | `600` | Maximum number of evaluation instances |
| `-n`, `--num-threads` | `1` | Number of parallel evaluation threads |
| `--display-name` | Same as `--model-name` | Display name for model metadata |
| `--creator` | Empty | Creator organization name |

#### Client selection logic

- **`--api-model` provided** → `fireworks_client.FireworksNoThinkingClient` (disables thinking, adds Arabic MCQ system prompt)
- **`--api-model` omitted** → `helm.clients.openai_client.OpenAIClient` (standard OpenAI-compatible client)

### `store_helm_results.py` — Standalone result storage

Store results from a previous HELM run without re-evaluating:

```bash
python store_helm_results.py \
  --run-dir benchmark_output/runs/<suite>/<run-name> \
  --suite <suite-name>
```

| Flag | Description |
|------|-------------|
| `--run-dir` | Path to a HELM run output directory containing `run_spec.json`, `scenario_state.json`, `per_instance_stats.json`, and `stats.json` |
| `--suite` | Suite label for database tracking |

---

## Available Benchmarks

| Benchmark | Description | Default Args | Run Spec Format |
|-----------|-------------|--------------|-----------------|
| `aratrust` | Arabic trustworthiness evaluation — safety, bias, toxicity, ethics, and more | `category=all` | `aratrust:category=all,model=<model>` |
| `arabic_mmlu` | Arabic Massive Multitask Language Understanding | `subset=all` | `arabic_mmlu:subset=all,model=<model>` |
| `alghafa` | AlGhafa Arabic NLP benchmark suite (multiple subtasks) | `subset=all` | `alghafa:subset=all,model=<model>` |
| `arabic_exams` | Arabic school exam questions | `subject=all` | `arabic_exams:subject=all,model=<model>` |
| `arabic_mmmlu` | Arabic Multilingual MMLU | `subject=all` | `arabic_mmmlu:subject=all,model=<model>` |

### Overriding benchmark arguments

Use `--benchmark-args` to run specific subsets:

```bash
# Only run the ethics category of AraTrust
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --benchmark aratrust \
  --benchmark-args "category=ethics" \
  --suite ethics-test \
  --max-instances 100

# Run a specific AlGhafa subset
python helm_eval.py \
  --model-name local/qwen3-5-9b \
  --api-base http://127.0.0.1:1234/v1 \
  --api-key lm-studio \
  --benchmark alghafa \
  --benchmark-args "subset=mcq_exams_test_merged" \
  --suite alghafa-test \
  --max-instances 50
```

---

## Currently Configured Models

| Model | Provider | Identifier | Tokenizer | Thinking | Client |
|-------|----------|------------|-----------|----------|--------|
| Qwen3.5 9B | Local (LM Studio) | `local/qwen3-5-9b` | `Qwen/Qwen3.5-9B` | Disabled at server level | `OpenAIClient` |
| Kimi K2.5 | Fireworks AI | `fireworks/kimi-k2p5` | `Qwen/Qwen2.5-7B` | Disabled via `reasoning_effort: "off"` | `FireworksNoThinkingClient` |

---

## Adding a New Model

You can either use the CLI (recommended) or manually edit the config files.

### Option A: Via CLI (recommended)

Just run `helm_eval.py` with the new model's flags — the YAML configs are created/updated automatically:

```bash
python helm_eval.py \
  --model-name fireworks/deepseek-r1 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/deepseek-r1 \
  --tokenizer deepseek-ai/DeepSeek-R1 \
  --display-name "DeepSeek R1 (Fireworks)" \
  --creator "DeepSeek" \
  --benchmark aratrust \
  --suite deepseek-test \
  --max-instances 10
```

### Option B: Manual configuration

#### 1. Add deployment in `model_deployments.yaml`

```yaml
- name: provider/model-name
  model_name: provider/model-name
  tokenizer_name: provider/model-name
  max_sequence_length: 131072
  client_spec:
    class_name: "fireworks_client.FireworksNoThinkingClient"  # or helm.clients.openai_client.OpenAIClient
    args:
      base_url: "https://api.provider.com/v1"
      openai_model_name: "accounts/provider/models/model-name"
```

For local models using the standard OpenAI client:

```yaml
- name: local/model-name
  model_name: local/model-name
  tokenizer_name: org/model-name
  max_sequence_length: 131072
  client_spec:
    class_name: "helm.clients.openai_client.OpenAIClient"
    args:
      api_key: "lm-studio"
      org_id: ""
      base_url: "http://127.0.0.1:1234/v1"
```

#### 2. Add metadata in `model_metadata.yaml`

```yaml
- name: provider/model-name
  display_name: Model Name (Provider)
  description: Description of the model
  creator_organization_name: Creator
  access: open
  num_parameters: 9000000000
  release_date: 2025-01-01
```

#### 3. Add tokenizer in `tokenizer_configs.yaml`

```yaml
- name: provider/model-name
  tokenizer_spec:
    class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
    args:
      pretrained_model_name_or_path: HuggingFaceOrg/ModelName
  end_of_text_token: "<|im_end|>"
  prefix_token: "<|im_start|>"
```

#### 4. Add API key in `credentials.conf`

```
providerApiKey: "your-api-key"
```

#### 5. Create a run spec `.conf` file

```
entries: [
  {description: "aratrust:category=all,model=provider/model-name", priority: 1}
]
```

---

## Configuration Files

### `model_deployments.yaml`

Defines how HELM connects to each model. Each entry specifies:

| Field | Description |
|-------|-------------|
| `name` | Unique model identifier (e.g. `fireworks/kimi-k2p5`) |
| `model_name` | HELM internal model name (usually same as `name`) |
| `tokenizer_name` | References an entry in `tokenizer_configs.yaml` |
| `max_sequence_length` | Maximum context window size |
| `client_spec.class_name` | Python class that handles API communication |
| `client_spec.args` | Client-specific arguments (base URL, API key, model name) |

### `model_metadata.yaml`

Display information for HELM's reporting UI:

| Field | Description |
|-------|-------------|
| `name` | Must match the deployment name |
| `display_name` | Human-readable name shown in reports |
| `description` | Model description |
| `creator_organization_name` | Who made the model |
| `access` | Access level (`open`, `limited`, `closed`) |
| `num_parameters` | Parameter count (optional) |
| `release_date` | Model release date (optional) |

### `tokenizer_configs.yaml`

Maps model names to HuggingFace tokenizers:

| Field | Description |
|-------|-------------|
| `name` | Must match the `tokenizer_name` in deployments |
| `tokenizer_spec.class_name` | Always `helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer` |
| `tokenizer_spec.args.pretrained_model_name_or_path` | HuggingFace model ID for the tokenizer |
| `end_of_text_token` | End-of-text token for the tokenizer |
| `prefix_token` | Prefix/start token for the tokenizer |

### `credentials.conf`

API keys in HELM's format. Key names follow the convention `<provider>ApiKey`:

```
openaiApiKey: "lm-studio"
fireworksApiKey: "fw_..."
```

### `run_specs_test.conf`

Example benchmark run specification (checked into git):

```
entries: [
  {description: "aratrust:category=all,model=fireworks/kimi-k2p5", priority: 1}
]
```

Generated `run_specs_<suite>.conf` files are gitignored and cleaned up automatically.

---

## Database Storage

Evaluation results are automatically stored in PostgreSQL when using `helm_eval.py`. The database is shared with the broader EduBench evaluation platform.

### Setup

Create a `.env` file in the project root:

```
DB_HOST=your-host
DB_NAME=your-database
DB_USER=your-user
DB_PASSWORD=your-password
DB_PORT=5432
```

### Tables used

#### `eval_runs` — One row per evaluation run

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Primary key (auto-generated) |
| `model_name` | text | Model identifier (e.g. `fireworks/kimi-k2p5`) |
| `task_name` | text | Benchmark name (e.g. `aratrust`, `arabic_mmlu`) |
| `judge_model` | text | Always `helm_exact_match` (HELM uses auto-grading) |
| `status` | text | Always `completed` (failed runs are not stored) |
| `started_at` | timestamp | Run start time |
| `completed_at` | timestamp | Run completion time |
| `total_samples` | integer | Number of samples evaluated |
| `metrics` | jsonb | Aggregated metrics (`exact_match`, `quasi_exact_match`, `prefix_exact_match`) |
| `config` | jsonb | Full run config including `suite`, `git_commit`, `git_branch`, `adapter_spec`, `scenario_spec` |
| `is_thinking_model` | boolean | Whether thinking/reasoning mode was detected |

#### `eval_samples` — One row per evaluation instance

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Primary key (auto-generated) |
| `run_id` | integer | Foreign key to `eval_runs` |
| `sample_index` | integer | Index within the run |
| `format` | text | Question format (e.g. `MCQ_4`, `MCQ_3`, `MCQ_2`) |
| `instruction` | text | Question text (Arabic) |
| `reference` | text | Correct answer |
| `prediction` | text | Model's extracted answer (mapped via output_mapping) |
| `raw_prompt` | text | Full prompt sent to the model |
| `raw_response` | text | Raw model response text |
| `judge_score` | numeric | `1.0` for correct, `0.0` for incorrect (exact_match) |
| `latency_ms` | numeric | Response latency in milliseconds |
| `answer_tokens` | integer | Token count for the model's response |
| `meta` | jsonb | Additional metadata (output_mapping, all references, finish_reason, instance_id) |

### How streaming works

Results are inserted in **batches of 10** during storage. The `scenario_state.json` file (which can be very large) is read using `ijson` streaming to avoid loading the entire file into memory.

Git commit SHA and branch name are captured at storage time for full reproducibility.

### Viewing results

After evaluation, use HELM's built-in tools:

```bash
# Summarize and generate reports
helm-summarize --suite <suite-name>

# Launch interactive web UI
helm-server --suite <suite-name>
```

See [docs/database-schema.md](docs/database-schema.md) for the full database schema including all EduBench tables.

---

## Disabling Thinking Mode

The [HELM Arabic leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html) requires thinking mode to be disabled for fair comparison. Many recent models (Kimi K2.5, Qwen3, DeepSeek R1) activate chain-of-thought reasoning by default, which:

1. **Corrupts answer parsing** — HELM expects a clean answer letter, but thinking models produce long reasoning chains that confuse the regex extractor
2. **Gives unfair extra compute** — thinking models use significantly more tokens, making comparisons inequitable
3. **Inflates latency** — reasoning tokens add substantial response time

### How we disable it

| Model Type | Method |
|------------|--------|
| **Local (LM Studio)** | Disable thinking at the server level in LM Studio settings |
| **Fireworks AI** | `FireworksNoThinkingClient` injects `reasoning_effort: "off"` into every request |

### Why HELM's built-in support isn't enough

HELM's `OpenAIClient` only sends `reasoning_effort` for OpenAI-specific model patterns (`o1`, `o3`, `gpt-5`). For other providers hosting thinking models, the parameter is never set — so the model thinks by default. Our custom client solves this.

### Provider-specific methods

For Fireworks AI:

| Method | Works? |
|--------|--------|
| `reasoning_effort: "off"` | Yes |
| `thinking: {"type": "disabled"}` | Yes |
| No parameter (default) | No — model still thinks |

---

## Custom Client: FireworksNoThinkingClient

The `fireworks_client.py` module extends HELM's `OpenAIClient` with two modifications:

### 1. Disable thinking mode

```python
raw_request["reasoning_effort"] = "off"
```

This is injected into every chat completion request, ensuring thinking models don't activate chain-of-thought reasoning.

### 2. Arabic MCQ system prompt

```python
_MCQ_SYSTEM_PROMPT = (
    "أجب عن أسئلة الاختيار من متعدد بحرف الإجابة فقط (أ، ب، ج، د، هـ) دون أي شرح أو تكرار للسؤال."
)
```

Translation: *"Answer multiple-choice questions with only the answer letter (A, B, C, D, E) without any explanation or repeating the question."*

This system prompt is prepended to every request that doesn't already have a system message. Without it, some models repeat the full question in Arabic, and HELM's regex extractor matches letters inside the repeated Arabic words instead of the actual answer letter.

### Full implementation

```python
class FireworksNoThinkingClient(OpenAIClient):
    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"

        messages: List[Dict[str, Any]] = raw_request.get("messages", [])
        if messages and not any(m.get("role") == "system" for m in messages):
            raw_request["messages"] = [
                {"role": "system", "content": _MCQ_SYSTEM_PROMPT}
            ] + messages

        return raw_request
```

---

## Advanced: Raw HELM Commands

For power users who want to run HELM directly. All commands require `PYTHONPATH=.` for the custom client to be discoverable.

```bash
source helm-env/bin/activate

# Run benchmark from a conf file
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite <suite-name> \
  --local-path . \
  --max-eval-instances <number>

# Run with parallel threads
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite <suite-name> \
  --local-path . \
  --max-eval-instances <number> \
  -n <threads>

# Summarize results into reports
helm-summarize --suite <suite-name>

# Launch interactive results web UI
helm-server --suite <suite-name>

# Generate comparison plots
helm-create-plots --suite <suite-name>
```

### Output directory structure

HELM writes all output to `benchmark_output/`:

```
benchmark_output/
└── runs/
    └── <suite-name>/
        └── <run-name>/
            ├── run_spec.json           # Run configuration
            ├── scenario_state.json     # Full evaluation state (can be very large)
            ├── per_instance_stats.json # Per-instance metrics
            ├── stats.json              # Aggregated metrics
            └── scenario.json           # Scenario definition
```

---

## Troubleshooting

### HELM can't find `fireworks_client`

Make sure `PYTHONPATH=.` is set. The `helm_eval.py` CLI sets this automatically, but raw `helm-run` commands need it explicitly:

```bash
PYTHONPATH=. helm-run --conf-paths run_specs_test.conf --suite test --local-path .
```

### Tokenizer download fails

The HuggingFace tokenizer downloads on first use. If you're behind a firewall or have no internet access, pre-download the tokenizer:

```python
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
```

### Model returns empty responses

- **LM Studio**: Ensure the model is loaded and the server is running at the configured port
- **Fireworks AI**: Check that the API key in `credentials.conf` is valid and the model identifier in `--api-model` is correct

### Low accuracy / garbled answers

- **Thinking mode not disabled**: Verify the model is using `FireworksNoThinkingClient` (check `model_deployments.yaml` for `client_spec.class_name`)
- **Wrong tokenizer**: Mismatched tokenizers can cause garbled text. Ensure `--tokenizer` matches the model family
- **Answer extraction failures**: Check the raw responses in the database (`eval_samples.raw_response`) — if the model is producing long explanations instead of answer letters, the Arabic MCQ system prompt may not be reaching the model

### Database connection fails

Verify `.env` credentials:

```bash
# Test connection
python -c "from store_helm_results import get_db_connection; get_db_connection(); print('OK')"
```

### Evaluation runs out of memory

Large benchmarks like AlGhafa can produce very large `scenario_state.json` files. The `store_helm_results.py` script uses `ijson` streaming to handle this, but `helm-run` itself may need more memory. Consider:

- Reducing `--max-instances`
- Running subsets individually with `--benchmark-args`

---

## Project Structure

```
arabic-helm-eval/
├── helm_eval.py              # Unified CLI: configure, evaluate, and store results
├── store_helm_results.py     # Standalone DB storage for HELM output (streaming with ijson)
├── fireworks_client.py       # Custom HELM client: disables thinking + Arabic MCQ prompt
├── model_deployments.yaml    # Model endpoints (API URL, client class, tokenizer)
├── model_metadata.yaml       # Model display info (name, creator, access level)
├── tokenizer_configs.yaml    # Tokenizer definitions (HuggingFace model mappings)
├── credentials.conf          # API keys (gitignored)
├── .env                      # Database credentials (gitignored)
├── run_specs_test.conf       # Example benchmark run specification
├── docs/
│   └── database-schema.md    # Full EduBench database schema (20 tables)
├── CLAUDE.md                 # Claude Code project instructions
├── .gitignore                # Ignores helm-env/, benchmark_output/, credentials, generated confs
└── helm-env/                 # Python 3.10 venv with HELM installed (not in git)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      helm_eval.py                       │
│              (Unified CLI orchestrator)                  │
├─────────────┬──────────────┬───────────────┬────────────┤
│  Configure  │  Generate    │  Run HELM     │  Store     │
│  YAML files │  run spec    │  subprocess   │  results   │
└──────┬──────┴──────┬───────┴───────┬───────┴─────┬──────┘
       │             │               │             │
       ▼             ▼               ▼             ▼
  model_*.yaml   run_specs_    helm-run with    PostgreSQL
  tokenizer_*    <suite>.conf  PYTHONPATH=.     (eval_runs +
  credentials                       │           eval_samples)
                                    │
                              ┌─────┴─────┐
                              │   HELM    │
                              │ Framework │
                              └─────┬─────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    OpenAIClient    FireworksNoThinkingClient
                    (local models)  (cloud models, no thinking)
                         │                     │
                         ▼                     ▼
                    LM Studio API       Fireworks AI API
                    localhost:1234      api.fireworks.ai
```

The project is intentionally a **thin layer on top of HELM**. The actual evaluation framework code lives inside `helm-env/lib/python3.10/site-packages/helm/`. Custom work here focuses on:

1. **Model configuration** — defining which models to evaluate and how to connect to them
2. **Thinking mode suppression** — ensuring fair comparison by disabling chain-of-thought reasoning
3. **Arabic-specific prompting** — guiding models to output clean answer letters for MCQ benchmarks
4. **Result persistence** — streaming evaluation data to a shared PostgreSQL database with git traceability
