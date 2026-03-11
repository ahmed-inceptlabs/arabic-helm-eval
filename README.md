# Arabic HELM Evaluation

Evaluate Arabic language models on standardized benchmarks using Stanford's [HELM](https://github.com/stanford-crfm/helm) framework. One command to configure, evaluate, and store results — supports local models via [LM Studio](https://lmstudio.ai/) and cloud providers like [Fireworks AI](https://fireworks.ai/).

Built for the [HELM Arabic Leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html).

---

## Quick Start

### 1. Install

```bash
python3.10 -m venv helm-env
source helm-env/bin/activate
pip install crfm-helm psycopg2-binary python-dotenv ijson pyyaml
git clone https://github.com/ahmed-inceptlabs/arabic-helm-eval.git
cd arabic-helm-eval
```

### 2. Configure credentials

Create `credentials.conf` in the project root:

```
openaiApiKey: "lm-studio"
fireworksApiKey: "your-fireworks-api-key"
```

### 3. Run an evaluation

```bash
source helm-env/bin/activate

# Fireworks AI model
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --benchmark aratrust \
  --suite my-experiment

# Local LM Studio model
python helm_eval.py \
  --model-name local/qwen3-5-9b \
  --api-base http://127.0.0.1:1234/v1 \
  --api-key lm-studio \
  --tokenizer Qwen/Qwen3.5-9B \
  --benchmark arabic_mmlu \
  --suite local-test \
  --max-instances 10

# Multiple benchmarks at once
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --benchmark aratrust arabic_mmlu alghafa arabic_exams arabic_mmmlu \
  --suite full-eval

# Parallel threads
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --benchmark aratrust arabic_mmlu \
  --suite base-kimi-k2p5 \
  -n 8
```

### 4. View results

```bash
helm-summarize --suite my-experiment
helm-server --suite my-experiment     # launches web UI
```

---

## How It Works

`helm_eval.py` is a single-command orchestrator that:

1. **Auto-configures YAML files** — writes model deployment, tokenizer, and metadata entries from your CLI flags
2. **Runs HELM** — generates a temporary run spec, calls `helm-run`, then deletes the run spec
3. **Stores results** — streams evaluation data to PostgreSQL in batches of 10

When running multiple benchmarks, steps 1-2 repeat for each benchmark. Results are stored once at the end.

### Auto-Generated Config Files

You never need to manually edit YAML files. `helm_eval.py` manages them automatically using an **upsert pattern**: it reads the existing file, finds the entry by model name, updates it if it exists or appends it if new. The rest of the file is left untouched.

| File | What it does | When updated |
|------|-------------|--------------|
| `model_deployments.yaml` | Model endpoints, client class, system prompt | **Before each benchmark** (see below) |
| `tokenizer_configs.yaml` | HuggingFace tokenizer mappings | Once per model setup |
| `model_metadata.yaml` | Display name, creator, access level | Once per model setup |
| `credentials.conf` | API keys | Once, if `--api-key` provided |
| `run_specs_<suite>.conf` | Benchmark entries for `helm-run` | Created before each run, **deleted after** |

### Per-Benchmark Reconfiguration

The key detail: `model_deployments.yaml` is **rewritten before every benchmark run**, not just once. This is because the system prompt must change depending on the benchmark type:

- **MCQ benchmarks** (aratrust, arabic_mmlu, alghafa, arabic_exams, arabic_mmmlu): No explicit `system_prompt` in the deployment config. The `FireworksNoThinkingClient` injects an Arabic MCQ prompt at runtime that tells the model to answer with just the letter (A, B, C, D).

- **Generation benchmarks** (alrage): The deployment is rewritten with `system_prompt: ""`, which tells the client to skip the MCQ prompt since the task is open-ended generation scored by GPT-4o.

```
For each benchmark:
  1. Update model_deployments.yaml (set system_prompt for MCQ vs generation)
  2. Generate run_specs_<suite>.conf with benchmark entries
  3. Run helm-run
  4. Delete run_specs file
After all benchmarks:
  5. Store results to PostgreSQL
```

### Architecture

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

---

## CLI Reference

### `helm_eval.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | **required** | Model identifier (`provider/name` format) |
| `--api-base` | **required** | API base URL |
| `--suite` | **required** | Suite name for output directory and DB tracking |
| `--api-model` | `None` | OpenAI model name — when set, uses `FireworksNoThinkingClient`; when omitted, uses standard `OpenAIClient` |
| `--api-key` | `None` | API key (uses existing `credentials.conf` value if not set) |
| `--tokenizer` | `Qwen/Qwen2.5-7B` | HuggingFace tokenizer identifier |
| `--max-seq-len` | `131072` | Maximum sequence length |
| `--benchmark` | `aratrust` | Benchmark(s) to run — pass multiple names to run several sequentially |
| `--benchmark-args` | per-benchmark default | Override benchmark arguments (e.g. `category=ethics`) |
| `--max-instances` | `600` | Maximum evaluation instances |
| `-n` | `1` | Parallel evaluation threads |
| `--display-name` | same as `--model-name` | Display name for reports |
| `--creator` | empty | Creator organization name |

### `store_helm_results.py` (standalone)

Store results from a previous HELM run without re-evaluating:

```bash
python store_helm_results.py --run-dir benchmark_output/runs/<suite>/<run-name> --suite <suite>
```

---

## Available Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| `aratrust` | MCQ | Arabic trustworthiness — safety, bias, toxicity, ethics |
| `arabic_mmlu` | MCQ | Arabic Massive Multitask Language Understanding |
| `alghafa` | MCQ | AlGhafa Arabic NLP suite (9 subtasks, auto-expanded) |
| `arabic_exams` | MCQ | Arabic school exam questions |
| `arabic_mmmlu` | MCQ | Arabic Multilingual MMLU (52 subjects, auto-expanded) |
| `alrage` | Generation | Arabic RAG evaluation, scored by GPT-4o |

> **Note:** `alrage` requires a valid OpenAI API key in `credentials.conf` (`openaiApiKey: "sk-..."`).

Override default benchmark arguments with `--benchmark-args`:

```bash
# Only run the ethics category of AraTrust
python helm_eval.py ... --benchmark aratrust --benchmark-args "category=ethics"

# Run a specific AlGhafa subset
python helm_eval.py ... --benchmark alghafa --benchmark-args "subset=mcq_exams_test_ar"
```

---

## Adding a New Model

Just run `helm_eval.py` with the new model's flags — all YAML configs are created automatically:

```bash
python helm_eval.py \
  --model-name fireworks/deepseek-r1 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/deepseek-r1 \
  --tokenizer deepseek-ai/DeepSeek-R1 \
  --display-name "DeepSeek R1" \
  --creator "DeepSeek" \
  --benchmark aratrust \
  --suite deepseek-test \
  --max-instances 10
```

This adds entries to `model_deployments.yaml`, `tokenizer_configs.yaml`, and `model_metadata.yaml`. If the model already exists, it updates the existing entries.

For manual configuration, see the existing YAML files for the expected format.

---

## Database Storage

### Setup

Create `.env` in the project root:

```
DB_HOST=your-host
DB_NAME=your-database
DB_USER=your-user
DB_PASSWORD=your-password
DB_PORT=5432
```

### Tables

- **`eval_runs`** — one row per evaluation run (model, benchmark, metrics, config, git commit/branch)
- **`eval_samples`** — one row per evaluation instance (question, reference answer, prediction, score, latency, raw prompt/response)

Results are streamed in batches of 10. Git commit SHA and branch are tracked for reproducibility. See [docs/database-schema.md](docs/database-schema.md) for the full schema.

---

## Disabling Thinking Mode

The [HELM Arabic leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html) requires thinking mode off for fair comparison. Thinking models produce reasoning chains that break HELM's answer extraction.

- **Local (LM Studio)**: Disable thinking in LM Studio settings
- **Fireworks AI**: `FireworksNoThinkingClient` (in `fireworks_client.py`) injects `reasoning_effort: "off"` into every request and prepends an Arabic system prompt instructing the model to respond with answer letters only

---

## Raw HELM Commands

For running HELM directly (requires `PYTHONPATH=.` for the custom client):

```bash
source helm-env/bin/activate

PYTHONPATH=. helm-run --conf-paths run_specs_test.conf --suite <suite> --local-path . --max-eval-instances <n>
helm-summarize --suite <suite>
helm-server --suite <suite>
helm-create-plots --suite <suite>
```

---

## Troubleshooting

- **`fireworks_client` not found**: Set `PYTHONPATH=.` before `helm-run`. The `helm_eval.py` CLI does this automatically.
- **Tokenizer download fails**: Pre-download with `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")`.
- **Empty responses**: Ensure the model is loaded (LM Studio) or the API key is valid (Fireworks).
- **Low accuracy / garbled answers**: Check that thinking mode is disabled and the tokenizer matches the model family. Inspect `eval_samples.raw_response` in the database.
- **Database connection fails**: Verify `.env` credentials with `python -c "from store_helm_results import get_db_connection; get_db_connection(); print('OK')"`.
- **Out of memory**: Reduce `--max-instances` or run subsets individually with `--benchmark-args`.

---

## Project Structure

```
arabic-helm-eval/
├── helm_eval.py              # Unified CLI: configure, evaluate, and store results
├── store_helm_results.py     # Standalone DB storage (streaming with ijson)
├── fireworks_client.py       # Custom client: disables thinking + Arabic MCQ prompt
├── model_deployments.yaml    # Auto-managed: model endpoints and client config
├── model_metadata.yaml       # Auto-managed: model display info
├── tokenizer_configs.yaml    # Auto-managed: HuggingFace tokenizer mappings
├── credentials.conf          # API keys (gitignored)
├── .env                      # Database credentials (gitignored)
├── run_specs_test.conf       # Example benchmark run specification
├── docs/
│   └── database-schema.md    # Full database schema
├── CLAUDE.md                 # Claude Code project instructions
├── .gitignore
└── helm-env/                 # Python 3.10 venv with HELM (not in git)
```
