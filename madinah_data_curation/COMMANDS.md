# Madinah Data Curation — CLI Commands

This is a concise cookbook for running the pipeline using plain `python3`.

## Setup

```bash
python3 -m venv madinah_data_curation/.venv
madinah_data_curation/.venv/bin/pip install datasets pyyaml openai pydantic faiss-cpu
```

If your home cache is full, set HF cache paths inside the repo:

```bash
export HF_HOME="madinah_data_curation/.hf_cache"
export HF_DATASETS_CACHE="madinah_data_curation/.hf_cache/datasets"
export HF_HUB_CACHE="madinah_data_curation/.hf_cache/hub"
```

## 1) Fetch datasets

```bash
python3 madinah_data_curation/00_fetch_datasets.py
```

CIDAR only (skip InstAr):
```bash
python3 madinah_data_curation/00_fetch_datasets.py --skip-instar
```

If the CIDAR filter matches nothing:
```bash
python3 madinah_data_curation/00_fetch_datasets.py --skip-instar --allow-cidar-fallback
```

## 2) Generate synthetic grammar data

OpenAI‑compatible API (default):
```bash
python3 madinah_data_curation/01_generate_synthetic_grammar.py \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --max-examples 1000
```

Fireworks API:
```bash
python3 madinah_data_curation/01_generate_synthetic_grammar.py \
  --api-base https://api.fireworks.ai/inference/v1 \
  --model accounts/fireworks/models/kimi-k2p5 \
  --max-examples 1000
```

For APIs that don't support structured output (e.g., some local models), use the
fallback mode which extracts JSON via regex:
```bash
python3 madinah_data_curation/01_generate_synthetic_grammar.py \
  --api-base http://127.0.0.1:1234/v1 \
  --model local-model \
  --max-examples 100 \
  --no-structured-output
```

## 3) Normalize + filter

```bash
python3 madinah_data_curation/02_normalize_filter.py
```

FastText language ID (optional):
```bash
python3 madinah_data_curation/02_normalize_filter.py --fasttext-model /path/to/lid.176.ftz
```

## 4) Format to ShareGPT

```bash
python3 madinah_data_curation/03_format_sharegpt.py
```

## 5) Build curriculum

```bash
python3 madinah_data_curation/07_build_curriculum.py
```

## 6) Diagnostics + audit

```bash
python3 madinah_data_curation/08_profile_report.py
python3 madinah_data_curation/09_sample_audit.py
```

If you want to use the venv explicitly, replace `python3` with:
```bash
madinah_data_curation/.venv/bin/python
```

---

## Single-run script (skips step 4 by default)

```bash
python3 madinah_data_curation/run_all.py \
  --synth-model gpt-4o-mini
```

If you want to include formatting:
```bash
python3 madinah_data_curation/run_all.py \
  --synth-model gpt-4o-mini \
  --run-format
```

Per-step arguments:
```bash
python3 madinah_data_curation/run_all.py \
  --fetch-skip-instar \
  --fetch-allow-cidar-fallback \
  --synth-api-base https://api.fireworks.ai/inference/v1 \
  --synth-model accounts/fireworks/models/kimi-k2p5 \
  --synth-max-examples 500 \
  --normalize-min-arabic-ratio 0.9
```
