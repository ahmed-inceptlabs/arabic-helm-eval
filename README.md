# Arabic HELM Evaluation

Evaluate Arabic language models on standardized benchmarks using Stanford's [HELM](https://github.com/stanford-crfm/helm) framework. Supports local models (LM Studio) and cloud providers (Fireworks AI), with automatic result storage in PostgreSQL.

## Prerequisites

- **Python 3.10** virtual environment with HELM installed (`helm-env/`)
- **API access**: Local model via [LM Studio](https://lmstudio.ai/) or a [Fireworks AI](https://fireworks.ai/) API key
- **PostgreSQL** (optional): For storing evaluation results. Credentials go in `.env`

## Quick Start

### 1. Activate the environment

```bash
source helm-env/bin/activate
```

### 2. Configure credentials

Create `credentials.conf` in the project root:

```
openaiApiKey: "lm-studio"
fireworksApiKey: "your-fireworks-api-key"
```

HELM resolves API keys by convention: a deployment named `fireworks/model-name` looks for `fireworksApiKey`.

### 3. Run an evaluation

The unified CLI configures HELM, runs the benchmark, and stores results in one command:

```bash
# Fireworks AI model
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --tokenizer Qwen/Qwen2.5-7B \
  --benchmark aratrust \
  --suite my-experiment \
  --max-instances 600

# Local LM Studio model
python helm_eval.py \
  --model-name local/qwen3-5-9b \
  --api-base http://127.0.0.1:1234/v1 \
  --api-key lm-studio \
  --tokenizer Qwen/Qwen3.5-9B \
  --benchmark arabic_mmlu \
  --suite local-test \
  --max-instances 10
```

This auto-generates YAML configs, runs HELM with `PYTHONPATH=.`, streams results to the database in batches, and cleans up temp files when done.

## Available Benchmarks

| Benchmark | Description | Default Args |
|-----------|-------------|--------------|
| `aratrust` | Arabic trustworthiness evaluation (safety, bias, etc.) | `category=all` |
| `arabic_mmlu` | Arabic Massive Multitask Language Understanding | `subset=all` |
| `alghafa` | AlGhafa Arabic NLP benchmark | `subset=all` |
| `arabic_exams` | Arabic school exam questions | `subject=all` |
| `arabic_mmmlu` | Arabic Multilingual MMLU | `subject=all` |

Pass `--benchmark-args` to override the default args (e.g. `--benchmark-args "category=ethics"`).

## Currently Configured Models

| Model | Provider | Thinking | Client |
|-------|----------|----------|--------|
| Qwen3.5 9B | Local (LM Studio) | Disabled at server level | `OpenAIClient` |
| Kimi K2.5 | Fireworks AI | Disabled via `reasoning_effort: "off"` | `FireworksNoThinkingClient` |

## Adding a New Model

### 1. Add deployment in `model_deployments.yaml`

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

### 2. Add metadata in `model_metadata.yaml`

```yaml
- name: provider/model-name
  display_name: Model Name (Provider)
  description: Description of the model
  creator_organization_name: Creator
  access: open
  release_date: 2025-01-01
```

### 3. Add tokenizer in `tokenizer_configs.yaml`

```yaml
- name: provider/model-name
  tokenizer_spec:
    class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
    args:
      pretrained_model_name_or_path: HuggingFaceOrg/ModelName
  end_of_text_token: "<|im_end|>"
  prefix_token: "<|im_start|>"
```

### 4. Add API key in `credentials.conf`

```
providerApiKey: "your-api-key"
```

### 5. Add run spec in a `.conf` file

```
entries: [
  {description: "aratrust:category=all,model=provider/model-name", priority: 1}
]
```

> **Tip:** `helm_eval.py` handles steps 1-5 automatically via CLI flags. Manual setup is only needed for custom configurations.

## Database Storage

Evaluation results are automatically stored in PostgreSQL when using `helm_eval.py`.

### Setup

Create a `.env` file in the project root:

```
DB_HOST=your-host
DB_NAME=your-database
DB_USER=your-user
DB_PASSWORD=your-password
DB_PORT=5432
```

### What gets stored

- **`eval_runs`** — One row per evaluation run (model, benchmark, suite, accuracy, git commit, timestamps)
- **`eval_samples`** — One row per evaluation instance (prompt, model response, correct answer, metrics)

Results are streamed in batches of 10 during evaluation. Git commit and branch are tracked for reproducibility.

### Standalone storage tool

To store results from a previous HELM run without re-evaluating:

```bash
python store_helm_results.py \
  --run-dir benchmark_output/runs/<suite>/<run-name> \
  --suite <suite-name>
```

See [docs/database-schema.md](docs/database-schema.md) for the full database schema.

## Disabling Thinking Mode

The [HELM Arabic leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html) requires thinking mode to be disabled for fair comparison. Many models (Kimi K2.5, Qwen3, DeepSeek R1) think by default, which corrupts HELM's answer parsing and gives unfair extra compute.

HELM's built-in `OpenAIClient` only sends `reasoning_effort` for OpenAI model patterns (`o1`, `o3`, `gpt-5`). For other providers, `fireworks_client.py` extends `OpenAIClient` to always inject `reasoning_effort: "off"`:

```python
class FireworksNoThinkingClient(OpenAIClient):
    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"
        return raw_request
```

For reference, Fireworks AI supports two methods to disable thinking:

| Method | Works? |
|--------|--------|
| `reasoning_effort: "off"` | Yes |
| `thinking: {"type": "disabled"}` | Yes |
| No parameter (default) | No — model still thinks |

## Advanced: Raw HELM Commands

For power users who want to run HELM directly. All commands require `PYTHONPATH=.` for the custom client.

```bash
source helm-env/bin/activate

# Run benchmark
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite <suite-name> \
  --local-path . \
  --max-eval-instances <number>

# Summarize results
helm-summarize --suite <suite-name>

# Launch results web UI
helm-server --suite <suite-name>

# Generate plots
helm-create-plots --suite <suite-name>
```

## Project Structure

```
arabic-helm-eval/
├── helm_eval.py              # Unified CLI: configure, evaluate, and store results
├── store_helm_results.py     # Standalone DB storage for HELM output
├── fireworks_client.py       # Custom HELM client that disables thinking mode
├── model_deployments.yaml    # Model endpoints (API URL, client class, tokenizer)
├── model_metadata.yaml       # Model display info (name, creator, access level)
├── tokenizer_configs.yaml    # Tokenizer definitions for each model
├── credentials.conf          # API keys (gitignored)
├── .env                      # Database credentials (gitignored)
├── run_specs_test.conf       # Example benchmark run specification
├── docs/
│   └── database-schema.md    # Full database schema documentation
├── CLAUDE.md                 # Claude Code project instructions
└── helm-env/                 # Python 3.10 venv with HELM installed
```
