# Commands Reference

All scripts run from the `semantic-data-dedup/` directory. Each accepts `--help` for full usage.

```bash
cd semantic-data-dedup
pip install openai faiss-cpu numpy sentence-transformers
```

---

## Quick Start: `run_pipeline.py`

Runs all six steps in one command. All per-step flags are available.

```bash
# Full pipeline with defaults (OpenAI embeddings, threshold 0.93)
python run_pipeline.py

# Custom threshold
python run_pipeline.py --threshold 0.96

# Local embeddings (no API key needed, slower)
python run_pipeline.py --backend local

# Re-cluster with a different threshold (skips expensive embedding + search)
python run_pipeline.py --start-from 4 --skip-embed --threshold 0.95

# Only run the cheap hash-based steps (no embeddings)
python run_pipeline.py --stop-after 3

# Custom input file
python run_pipeline.py --input /path/to/my_data.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `ar93_en7_mcq85_open15_cot96_211k.jsonl` | Source JSONL file |
| `--report-dir` | `reports/` | Reports output directory |
| `--data-dir` | `data/` | Data output directory |
| `--sample-size` | `50` | Rows per sample slice (Step 2) |
| `--seed` | `42` | Random seed (Step 2) |
| `--backend` | `openai` | Embedding backend: `openai` or `local` (Step 4) |
| `--threshold` | `0.93` | Cosine similarity threshold (Step 4) |
| `--top-k` | `10` | Nearest neighbors per row (Step 4) |
| `--skip-embed` | off | Reuse cached embeddings (Step 4) |
| `--start-from` | `1` | Start from step 1-6 |
| `--stop-after` | `6` | Stop after step 1-6 |

---

## Step 1: Profile the Dataset

Streams the JSONL and produces a stats report. Constant memory.

```bash
# Default: profile the source file
python 01_profile.py

# Custom input file
python 01_profile.py --input /path/to/other_dataset.jsonl

# Custom report directory
python 01_profile.py --report-dir /tmp/my_reports
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `ar93_en7_mcq85_open15_cot96_211k.jsonl` | Path to input JSONL |
| `--report-dir` | `reports/` | Where to write `01_profile_report.json` |

**Output:** `reports/01_profile_report.json`
**Runtime:** ~2 minutes for 211K rows

---

## Step 2: Stratified Sampling

Reservoir sampling to collect representative slices for manual inspection.

```bash
# Default: 50 rows per slice, seed=42
python 02_sample.py

# Larger samples
python 02_sample.py --sample-size 100

# Different seed for a fresh random draw
python 02_sample.py --seed 123

# All options
python 02_sample.py \
    --input ar93_en7_mcq85_open15_cot96_211k.jsonl \
    --output-dir reports/02_samples \
    --sample-size 50 \
    --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `ar93_en7_mcq85_open15_cot96_211k.jsonl` | Path to input JSONL |
| `--output-dir` | `reports/02_samples/` | Where to write sample JSONL files |
| `--sample-size` | `50` | Number of rows per sample slice |
| `--seed` | `42` | Random seed for reproducibility |

**Output:** `reports/02_samples/` (16 JSONL files + `index.json`)
**Runtime:** ~2 minutes

---

## Step 3: Exact Dedup

Hash-based deduplication at five normalization levels (L1–L5).

```bash
# Default
python 03_exact_dedup.py

# Custom paths
python 03_exact_dedup.py \
    --input ar93_en7_mcq85_open15_cot96_211k.jsonl \
    --report-dir reports \
    --data-dir data
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `ar93_en7_mcq85_open15_cot96_211k.jsonl` | Path to input JSONL |
| `--report-dir` | `reports/` | Where to write `03_exact_dedup_report.json` |
| `--data-dir` | `data/` | Where to write `03_deduped.jsonl` and `03_exact_dedup_ids.json` |

**Output:**
- `data/03_deduped.jsonl` — deduped at L3 (think-stripped pair)
- `data/03_exact_dedup_ids.json` — cluster membership for all levels
- `reports/03_exact_dedup_report.json`

**Runtime:** ~4 minutes (two passes over the file)

---

## Step 4: Semantic Dedup

Embedding-based near-duplicate detection using OpenAI API + FAISS.

```bash
# Default: OpenAI backend, threshold 0.93
python 04_semantic_dedup.py

# Explicit options
python 04_semantic_dedup.py --backend openai --threshold 0.93

# Stricter threshold (fewer removals, higher precision)
python 04_semantic_dedup.py --threshold 0.96

# Looser threshold (more removals, catches more paraphrases)
python 04_semantic_dedup.py --threshold 0.90

# More neighbors for better recall
python 04_semantic_dedup.py --top-k 20

# Local embedding model (no API key needed, but slow on CPU)
python 04_semantic_dedup.py --backend local

# Reuse cached embeddings with a different threshold
python 04_semantic_dedup.py --skip-embed --threshold 0.95
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `openai` | Embedding backend: `openai` or `local` |
| `--threshold` | `0.93` | Cosine similarity threshold for clustering |
| `--top-k` | `10` | Number of nearest neighbors to retrieve per row |
| `--skip-embed` | off | Skip embedding; load cached `data/04_embeddings.npy` |

**Output:**
- `data/04_embeddings.npy` — 211K x 1536 memmap file (1.23 GB)
- `data/04_semantic_clusters.json` — cluster membership + representatives
- `data/04_deduped.jsonl` — deduplicated output
- `reports/04_similarity_histogram.json` — similarity score distribution
- `reports/04_threshold_samples/` — sample pairs at each similarity band

**Runtime:**
- Embedding (OpenAI): ~15-20 min (API-bound)
- FAISS search: ~45-60 min on 2-core ARM (CPU-bound)
- Total: ~60-80 min

### Re-running with a different threshold

The most expensive part is embedding + FAISS search. To test different thresholds without re-embedding:

```bash
# First run: full pipeline
python 04_semantic_dedup.py --threshold 0.93

# Re-run: reuse embeddings, just re-cluster at a new threshold
python 04_semantic_dedup.py --skip-embed --threshold 0.96
```

**Note:** `--skip-embed` skips both embedding AND FAISS search, loading from cached files. If you want to re-search with different `--top-k`, delete `data/04_embeddings.npy` and run without `--skip-embed`.

---

## Step 5: Leakage Check

Heuristic pattern matching against the six HELM Arabic benchmarks.

```bash
# Default: reads from data/03_deduped.jsonl
python 05_leakage_check.py

# Custom input (e.g., check the semantic-deduped file instead)
python 05_leakage_check.py --input data/04_deduped.jsonl

# Custom report directory
python 05_leakage_check.py --report-dir /tmp/reports
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/03_deduped.jsonl` | Path to JSONL file to check |
| `--report-dir` | `reports/` | Where to write `05_leakage_report.json` |

**Output:** `reports/05_leakage_report.json`
**Runtime:** ~2-3 minutes

---

## Step 6: Build Clean Dataset

Combines all removal decisions into a final clean JSONL.

```bash
# Default
python 06_build_clean.py

# Custom paths
python 06_build_clean.py \
    --source ar93_en7_mcq85_open15_cot96_211k.jsonl \
    --report-dir reports \
    --data-dir data
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `ar93_en7_mcq85_open15_cot96_211k.jsonl` | Original source JSONL |
| `--report-dir` | `reports/` | Directory with reports from steps 1-5 |
| `--data-dir` | `data/` | Directory with data from steps 3-4 |

**Output:**
- `data/06_clean.jsonl` — final clean dataset
- `data/06_removal_log.jsonl` — every removed row with reason(s)
- `reports/06_final_report.md` — human-readable summary

**Runtime:** ~2 minutes

---

## Full Pipeline (copy-paste)

```bash
cd semantic-data-dedup

# Step 1: Profile
python 01_profile.py

# Step 2: Sample (can run in parallel with Step 3)
python 02_sample.py

# Step 3: Exact dedup
python 03_exact_dedup.py

# Step 4: Semantic dedup (longest step, ~60-80 min)
python 04_semantic_dedup.py --backend openai --threshold 0.93

# Step 5: Leakage check (can run in parallel with Step 4)
python 05_leakage_check.py

# Step 6: Build final clean dataset
python 06_build_clean.py
```

### Parallel execution

Steps 2 and 3 are independent of each other (both read the source file). Step 5 reads from `data/03_deduped.jsonl` so it can run as soon as Step 3 finishes, in parallel with Step 4:

```bash
# After Step 1:
python 02_sample.py &
python 03_exact_dedup.py
wait

# After Step 3:
python 04_semantic_dedup.py --backend openai --threshold 0.93 &
python 05_leakage_check.py
wait

# After Steps 4+5:
python 06_build_clean.py
```

---

## Checking help for any script

```bash
python 01_profile.py --help
python 02_sample.py --help
python 03_exact_dedup.py --help
python 04_semantic_dedup.py --help
python 05_leakage_check.py --help
python 06_build_clean.py --help
```
