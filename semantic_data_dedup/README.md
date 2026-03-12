# Semantic Data Dedup Pipeline

Six-step pipeline to profile, deduplicate, and clean a 211K-row Arabic fine-tuning JSONL corpus (`ar93_en7_mcq85_open15_cot96_211k.jsonl`). Designed for a 2-core ARM / 8 GB RAM machine with no GPU.

---

## Glossary

| Abbreviation | Meaning |
|:-------------|:--------|
| **L1** | Normalization Level 1 — **Raw pair**: SHA-256 hash of `user + assistant` text exactly as-is |
| **L2** | Normalization Level 2 — **Whitespace-normalized pair**: collapse all whitespace, then hash |
| **L3** | Normalization Level 3 — **Think-stripped pair**: remove `<think>...</think>` blocks from assistant text, normalize whitespace, then hash. *This is the level used for actual removal.* |
| **L4** | Normalization Level 4 — **User-only**: hash only the whitespace-normalized user prompt (ignores assistant response entirely) |
| **L5** | Normalization Level 5 — **User + Arabic canonicalization**: normalize alef variants (آ أ إ ٱ → ا), ya/alef-maksura (ى → ي), ta-marbuta (ة → ه), strip diacritics (tashkeel), unify punctuation, then hash user prompt |
| **MCQ** | Multiple-Choice Question |
| **CoT** | Chain-of-Thought — reasoning traces inside `<think>...</think>` tags in assistant responses |
| **FAISS** | Facebook AI Similarity Search — library for efficient vector similarity search |
| **IndexFlatIP** | FAISS exact inner-product index (equivalent to cosine similarity on L2-normalized vectors) |
| **Memmap** | Memory-mapped file — numpy array stored on disk, paged into RAM on demand by the OS |
| **Tashkeel** | Arabic diacritical marks (fatha, damma, kasra, etc.) — optional pronunciation guides |

---

## Pipeline Overview

```
  SOURCE (211,075 rows, 616 MB)
       │
       ▼
  ┌─────────────┐
  │ 01_profile   │ ── Streaming stats, language mix, MCQ detection
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────┐
│02_sample│ │03_exact_dedup│ ── Hash-based dedup at 5 levels (L1-L5)
└────────┘ └──────┬──────┘
                  │  211,058 rows (−17)
                  ▼
          ┌───────────────┐
          │04_semantic_dedup│ ── OpenAI embeddings + FAISS cosine search
          └──────┬────────┘
                 │  200,619 rows (−10,439)
    ┌────────────┤
    ▼            ▼
┌──────────┐ ┌──────────┐
│05_leakage│ │threshold  │ ── Manual review of similarity bands
│  _check  │ │calibration│
└────┬─────┘ └──────────┘
     │
     ▼
┌──────────────┐
│06_build_clean│ ── Combine all removals → final clean JSONL
└──────────────┘
     │
     ▼
  CLEAN (200,428 rows, −5.04%)
```

---

## Step 1: Streaming Profile (`01_profile.py`)

Reads the entire 211K JSONL line-by-line in constant memory. No data is loaded into RAM — everything is computed in a single pass.

**What it measures:**

| Metric | Value |
|--------|-------|
| Total rows | 211,075 |
| Parse failures | 0 |
| Empty user content | 0 |
| Empty assistant content | 0 |

### Language Distribution (user prompts)

```
Arabic  ████████████████████████████████████████████████░░  92.7%  (195,662)
English ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6.6%   (13,875)
Mixed   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.7%    (1,538)
```

### Question Type

```
Open-ended  ████████████████████████████████████████░░░░░░░░░░  76.7%  (161,875)
MCQ         ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  23.3%   (49,200)
```

> **Note:** The filename claims `mcq85` (85% MCQ), but the heuristic detector found only 23.3%. This is because many MCQ rows use non-standard formatting (numbered options, inline choices) that don't match the `أ) ب) ج)` / `A) B) C)` patterns.

### Chain-of-Thought (`<think>` tags)

```
With <think>   ████████████████████████████████████████████████░░  95.7%  (201,985)
Without        ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4.3%    (9,090)
```

Think block length: mean=1,011 chars, max=7,958 chars.

### Text Length Distributions

```
User prompt length (chars):     mean=666,   min=10,   max=19,922
Assistant response length:      mean=1,028, min=4,    max=13,265
```

Most user prompts are 100–400 chars. Assistant responses peak around 400–1,200 chars.

### Message Structure

| Pattern | Count |
|---------|-------|
| `user → assistant` | 196,159 (93%) |
| `system → user → assistant` | 14,916 (7%) |

### Arabic Normalization Stats

| Feature | Rows containing it |
|---------|-------------------|
| Alef variants (آ أ إ ٱ) | 197,676 (93.7%) |
| Ta marbuta (ة) | 197,676 (93.7%) |
| Ya variants (ى) | 190,727 (90.4%) |
| Tashkeel/diacritics | 193,391 (91.6%) |

**Output:** `reports/01_profile_report.json`

---

## Step 2: Stratified Sampling (`02_sample.py`)

Uses reservoir sampling to collect 50-row representative slices for manual quality inspection without loading the full dataset.

**16 sample slices collected:**

| Slice | Description | Rows |
|-------|-------------|------|
| `uniform_random` | Random across entire corpus | 50 |
| `lang_arabic` | Arabic-only user prompts | 50 |
| `lang_english` | English-only user prompts | 50 |
| `lang_mixed` | Mixed Arabic/English prompts | 50 |
| `type_mcq` | MCQ-detected questions | 50 |
| `type_open_ended` | Open-ended questions | 50 |
| `benchmark_alghafa_sentiment` | Sentiment analysis patterns | 50 |
| `benchmark_aratrust_safety` | Safety/ethics patterns | 50 |
| `benchmark_mmlu_subject` | MMLU subject patterns | 50 |
| `benchmark_exam_style` | Exam-style formatting | 50 |
| `benchmark_alrage_generation` | Generation task patterns | 50 |
| `longest_think` | Longest `<think>` blocks | 50 |
| `shortest_user_prompt` | Shortest user prompts | 50 |
| `longest_user_prompt` | Longest user prompts | 50 |
| `shortest_assistant` | Shortest assistant responses | 50 |

**Output:** `reports/02_samples/` (one JSONL per slice + `index.json`)

---

## Step 3: Exact Dedup (`03_exact_dedup.py`)

Hash-based deduplication at five normalization levels. Each row is hashed (SHA-256) at each level to find exact matches.

### Five Normalization Levels

```
L1  Raw pair (user+assistant as-is)          → 0 duplicates
L2  Whitespace-normalized pair               → 0 duplicates
L3  Think-stripped pair (remove <think>)      → 17 duplicates (0.01%)
L4  User-only (normalized)                   → 22 duplicates (0.01%)
L5  User + Arabic canonicalization           → 176 duplicates (0.08%)
    (alef/ya/ta-marbuta normalization,
     strip diacritics, unify punctuation)
```

**Key finding:** The dataset is remarkably clean at the exact-duplicate level. Only 17 rows are identical after stripping `<think>` blocks (L3). Even with aggressive Arabic canonicalization (L5), only 176 duplicates exist.

**L3 was chosen as the dedup level** — it removes rows that differ only in their `<think>` reasoning but have identical user prompts and final answers.

| Metric | Value |
|--------|-------|
| Input rows | 211,075 |
| Kept | 211,058 |
| Removed | 17 |

**Top repeated templates (L5):** Mostly AlGhafa-style sentiment MCQs with the same template but different short review texts inserted.

**Output:**
- `reports/03_exact_dedup_report.json`
- `data/03_deduped.jsonl` (211,058 rows — input for Step 4)
- `data/03_exact_dedup_ids.json` (cluster membership)

---

## Step 4: Semantic Dedup (`04_semantic_dedup.py`)

Finds *near*-duplicates that are semantically equivalent but textually different (e.g., paraphrases, reworded questions, Arabic normalization variants that slip past hashing).

### How It Works

```
   211,058 user prompts
          │
          ▼  Normalize: strip <think>, collapse whitespace, truncate >5K chars
          │
          ▼  Embed with OpenAI text-embedding-3-small (1536 dims)
          │   - Batch size: 200 texts per API call
          │   - Cost: ~$0.08 for 211K prompts
          │   - Checkpoint every 5K rows to memmap file
          │
          ▼  L2-normalize all vectors
          │
          ▼  FAISS IndexFlatIP (brute-force inner product = cosine similarity)
          │   - Top-10 nearest neighbors per row
          │   - 211K × 1536 × 4 bytes = 1.23 GB in memory
          │
          ▼  Similarity histogram + threshold calibration samples
          │   - Bands: 0.90–0.93, 0.93–0.96, 0.96–0.99, 0.99+
          │   - 50 sample pairs per band for manual review
          │
          ▼  Connected-component clustering (threshold=0.93)
          │   - Union-Find on edges above threshold
          │   - Keep representative with longest non-<think> assistant response
          │
          ▼  Write deduplicated output
```

### Results

| Metric | Value |
|--------|-------|
| Input rows | 211,058 |
| Clusters found | 3,508 |
| Rows in clusters | 13,947 |
| Removed | 10,439 (4.9%) |
| Kept | 200,619 |

### Similarity Distribution

```
Band           Count     Interpretation
─────────────────────────────────────────────
0.00 – 0.50      5,968   Unrelated pairs
0.50 – 0.70     36,812   Different topic
0.70 – 0.80     84,033   Same domain, different question
0.80 – 0.85     40,265   Related questions
0.85 – 0.90     22,214   Similar phrasing
0.90 – 0.93      7,819   Near-duplicates (below threshold)
0.93 – 0.95      3,812   ← removed (above threshold)
0.95 – 0.96      1,622   ← removed
0.96 – 0.97      1,760   ← removed
0.97 – 0.98      1,898   ← removed
0.98 – 0.99      2,151   ← removed
0.99 – 1.00      2,704   ← removed (near-identical)
```

Stats: mean=0.77, median=0.78, p95=0.95, p99=0.99

### Memory Optimization
- **Embeddings** are written directly to a numpy memmap file (1.23 GB on disk, not in Python memory)
- **Full rows** are never loaded into memory — streamed from disk when needed
- **Only text strings** live in RAM (~140 MB for 211K prompts)
- **Checkpoint resume** — if the process crashes, it picks up from the last completed batch

### Embedding Model Choice

| Model | Dims | Arabic Quality | Max Tokens | Cost (211K prompts) |
|-------|------|---------------|------------|---------------------|
| **text-embedding-3-small** (chosen) | 1536 | Strong | 8,192 | ~$0.08 |
| text-embedding-3-large | 3072 | Stronger | 8,192 | ~$0.26 |
| intfloat/multilingual-e5-base (local fallback) | 768 | Good | 512 | Free (3-6 hrs CPU) |

Texts exceeding 5,000 chars are truncated before embedding. Arabic tokenizes at ~1.5 chars/token with cl100k_base, so 5K chars ≈ 3.3K tokens (well under the 8,192 limit).

**Output:**
- `data/04_embeddings.npy` (1.23 GB memmap)
- `reports/04_similarity_histogram.json`
- `reports/04_threshold_samples/` (sample pairs at each similarity band)
- `data/04_semantic_clusters.json`
- `data/04_deduped.jsonl` (200,619 rows)

---

## Step 5: Leakage Check (`05_leakage_check.py`)

Heuristic pattern matching to flag rows that may derive from the six HELM Arabic benchmarks. This estimates contamination risk, not confirmed leakage.

### Detection Signals

1. **Structural fingerprints:** MCQ option formatting, true/false patterns, sentiment scales
2. **Keyword anchors:** Benchmark-specific terms (subject names, Arabic safety terms, exam vocabulary)
3. **Direct benchmark mentions:** Rows containing "HELM", "benchmark", "leaderboard", benchmark names
4. **Answer-pattern leakage:** Assistant text with scoring rubrics or "الإجابة الصحيحة هي:" wrappers

### Results

```
Benchmark          Flagged    %
─────────────────────────────────
alghafa            128,644   61.0%  ← Mostly sentiment MCQs with "رأي سلبي/إيجابي"
arabic_exams        37,849   17.9%  ← Exam-style keywords (امتحان, اختبار)
arabic_mmlu         26,692   12.7%  ← Subject keywords (biology, law, etc.)
aratrust             3,963    1.9%  ← Safety/ethics keywords
arabic_mmmlu             4    0.0%
alrage                   8    0.0%
─────────────────────────────────
Total unique       167,268   79.3%  (heuristic, many false positives)
```

**Higher-confidence signals:**
- Direct benchmark mentions: **182 rows** (names like "HELM", "benchmark")
- Rubric leak rows: **31,914 rows** (contain "الإجابة الصحيحة هي:" in assistant text)

> **Important:** The 79.3% heuristic flag rate is inflated. Most flagged rows are *legitimately about these topics* (e.g., a question about biology naturally triggers the MMLU subject pattern). The high-confidence signals (182 direct mentions + 31,914 rubric leaks) are more actionable.

**Output:** `reports/05_leakage_report.json`

---

## Step 6: Build Clean Dataset (`06_build_clean.py`)

Combines removal decisions from all previous steps into a final clean JSONL.

### Removal Breakdown

| Reason | Rows Removed |
|--------|-------------|
| Semantic duplicate (cosine >= 0.93) | 10,439 |
| Leakage high-confidence (benchmark mentions + rubric patterns) | 200 |
| Exact duplicate (L3 think-stripped) | 17 |
| **Total removed** | **10,647 (5.04%)** |

### Final Result

```
Source:   211,075 rows  (616 MB)
Clean:   200,428 rows  (594 MB)
Removed:  10,647 rows  (5.04%)
```

### Outputs

| File | Description |
|------|-------------|
| `data/06_clean.jsonl` | Final deduplicated dataset (200,428 rows) |
| `data/06_removal_log.jsonl` | Every removed row with removal reason(s) |
| `reports/06_final_report.md` | Human-readable summary with all stats |

---

## File Structure

```
semantic-data-dedup/
├── ar93_en7_mcq85_open15_cot96_211k.jsonl  # Source (untouched, 616 MB)
├── requirements.txt                         # openai, faiss-cpu, numpy, sentence-transformers
├── run_pipeline.py                          # Single-command runner for all steps
├── 01_profile.py                            # Streaming profiler
├── 02_sample.py                             # Stratified reservoir sampling
├── 03_exact_dedup.py                        # 5-level hash dedup
├── 04_semantic_dedup.py                     # Embedding + FAISS dedup
├── 05_leakage_check.py                      # Benchmark contamination check
├── 06_build_clean.py                        # Final assembly
├── README.md                                # Pipeline overview + results
├── detailed_readme.md                       # Algorithmic deep dive
├── COMMANDS.md                              # CLI reference for every script
├── data/
│   ├── 03_deduped.jsonl                     # After exact dedup (211,058 rows)
│   ├── 03_exact_dedup_ids.json              # Duplicate cluster IDs
│   ├── 04_embeddings.npy                    # 211K × 1536 memmap (1.23 GB)
│   ├── 04_embeddings_checkpoint.npy         # Partial embedding checkpoint
│   ├── 04_semantic_clusters.json            # Semantic duplicate clusters (3,508)
│   ├── 04_deduped.jsonl                     # After semantic dedup (200,619 rows)
│   ├── 06_clean.jsonl                       # Final clean dataset (200,428 rows)
│   └── 06_removal_log.jsonl                 # Removal audit log (10,647 entries)
└── reports/
    ├── 01_profile_report.json
    ├── 02_samples/                          # 16 stratified sample slices
    ├── 03_exact_dedup_report.json
    ├── 04_similarity_histogram.json
    ├── 04_threshold_samples/                # Pairs at 4 similarity bands
    ├── 05_leakage_report.json
    └── 06_final_report.md                   # Human-readable summary
```

## Running

```bash
pip install openai faiss-cpu numpy sentence-transformers
```

### One command (recommended)

```bash
# Full pipeline with defaults
python run_pipeline.py

# Custom threshold
python run_pipeline.py --threshold 0.96

# Local embeddings (no API key needed)
python run_pipeline.py --backend local

# Re-cluster with a different threshold (reuses cached embeddings)
python run_pipeline.py --start-from 4 --skip-embed --threshold 0.95

# Only run the cheap steps (no embeddings)
python run_pipeline.py --stop-after 3
```

### Step by step

```bash
python 01_profile.py
python 02_sample.py
python 03_exact_dedup.py
python 04_semantic_dedup.py --backend openai --threshold 0.93
python 05_leakage_check.py
python 06_build_clean.py
```

Every script accepts `--help`. See `COMMANDS.md` for full flag reference.
