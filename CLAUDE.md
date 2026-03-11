# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Stanford's **HELM** (Holistic Evaluation of Language Models) framework to evaluate Arabic language models. Models are served locally via **LM Studio** and accessed through an OpenAI-compatible API at `http://127.0.0.1:1234/v1`.

## Environment Setup

- Python virtual environment: `helm-env/` (Python 3.10)
- Activate: `source helm-env/bin/activate`
- HELM is installed via pip (`crfm-helm` package) inside the venv

## Key Configuration Files

- `model_deployments.yaml` — Defines model endpoints (client class, API base URL, tokenizer)
- `model_metadata.yaml` — Model display info (name, parameters, creator, access level)
- `credentials.conf` — API keys (uses `"lm-studio"` as the OpenAI API key placeholder)

## Running Evaluations

All commands require the venv to be activated first. `PYTHONPATH=.` is required so HELM can find the custom `fireworks_client.py`.

```bash
# Run a benchmark evaluation
PYTHONPATH=. helm-run --conf-paths run_specs_test.conf --suite <suite-name> \
  --local-path . --max-eval-instances <number>

# Summarize results
helm-summarize --suite <suite-name>

# Launch results web UI
helm-server --suite <suite-name>

# Generate plots
helm-create-plots --suite <suite-name>
```

## Unified CLI (Recommended)

One command to configure, evaluate, and store results:

```bash
# Fireworks model (single benchmark)
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --tokenizer Qwen/Qwen2.5-7B \
  --benchmark aratrust \
  --suite my-experiment \
  --max-instances 600

# Multiple benchmarks in one command
python helm_eval.py \
  --model-name fireworks/kimi-k2p5 \
  --api-base https://api.fireworks.ai/inference/v1 \
  --api-model accounts/fireworks/models/kimi-k2p5 \
  --tokenizer Qwen/Qwen2.5-7B \
  --benchmark aratrust arabic_mmlu alghafa \
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

This auto-generates YAML configs, runs HELM, and streams results to `eval_runs` + `eval_samples` in batches of 10. Git commit is tracked per run. DB credentials in `.env`.

**Available benchmarks:** `aratrust`, `arabic_mmlu`, `alghafa`, `arabic_exams`, `arabic_mmmlu`, `alrage`, `madinah_qa` (pass multiple to `--benchmark` to run several at once)

> **Note:** `alrage` is a generation benchmark (not MCQ) scored by GPT-4o. Requires a valid OpenAI API key in `credentials.conf`.

### Standalone storage (alternative)

```bash
# Store results from a specific run directory
python store_helm_results.py --run-dir benchmark_output/runs/<suite>/<run-name> --suite <suite>
```

## Architecture

The project is a thin configuration layer on top of HELM. The actual evaluation framework code lives inside `helm-env/lib/python3.10/site-packages/helm/`. Custom work here focuses on:

1. Defining which models to evaluate (`model_deployments.yaml`)
2. Providing model metadata for reporting (`model_metadata.yaml`)
3. Configuring run specifications for Arabic-specific benchmarks
4. Custom client (`fireworks_client.py`) to disable thinking mode for Fireworks AI models

Models connect via `helm.clients.openai_client.OpenAIClient` (local) or `fireworks_client.FireworksNoThinkingClient` (Fireworks AI with thinking disabled).
