# Deduplication Pipeline: Algorithmic Deep Dive

This document explains the **why** behind every technique in the pipeline — the mathematical formulations, the design trade-offs, and what makes Arabic text different from English when it comes to deduplication.

---

## The Problem

We have a 211K-row JSONL corpus of `(user_prompt, assistant_response)` pairs used to fine-tune a model for Arabic HELM benchmarks. The dataset may contain:

1. **Exact duplicates** — identical rows (trivial)
2. **Near-exact duplicates** — same content with minor formatting differences (whitespace, diacritics, `<think>` traces)
3. **Semantic near-duplicates** — paraphrased or reworded versions of the same question
4. **Benchmark leakage** — training data that directly overlaps with evaluation benchmarks

Each type requires a different detection method. We layer them cheapest-first:

```
Cost:     $0                  $0                    ~$0.08              $0
Speed:    O(n)                O(n)                  O(n²)              O(n)
          ─────────────────────────────────────────────────────────────────►
          Hash-based dedup    Arabic normalization   Embedding + FAISS   Pattern matching
          (exact)             (near-exact)           (semantic)          (leakage)
```

---

## Why Layered Deduplication?

A single pass with embeddings could theoretically catch everything, but:

- **Embedding 211K texts costs money** (API calls) and compute (FAISS search is O(n² · d))
- **Hash-based dedup is free** and runs in O(n) — removing exact duplicates first makes the expensive step smaller
- **Different normalization levels reveal different insights** — knowing that L1 has 0 dupes but L5 has 176 tells us the dataset was already pre-cleaned but has Arabic normalization inconsistencies
- **Semantic dedup catches what hashing cannot** — "ما هي عاصمة مصر؟" and "أخبرني ما عاصمة جمهورية مصر العربية" are semantically identical but completely different strings

---

## Step 1: Streaming Profile — Why Single-Pass?

### The Constraint

The source JSONL is 616 MB. On an 8 GB machine, loading it all into memory would consume ~3-4 GB (JSON overhead), leaving too little for later steps. Instead, we stream line-by-line:

```python
for line in open(file):
    row = json.loads(line)
    # accumulate counters, never store row
```

Memory usage: O(1) regardless of dataset size. Only counters and a histogram live in RAM.

### Language Detection Without an LLM

We use Unicode script ranges rather than a language model:

```
Arabic:  U+0600–U+06FF, U+0750–U+077F, U+08A0–U+08FF, U+FB50–U+FDFF, U+FE70–U+FEFF
Latin:   A–Z, a–z
```

Classification rule:

```
ar_fraction = arabic_char_count / (arabic_char_count + latin_char_count)

if ar_fraction ≥ 0.85  → "arabic"
if ar_fraction ≤ 0.15  → "english"
else                    → "mixed"
```

**Why 0.85/0.15?** Arabic text routinely contains Latin characters (numbers, abbreviations, transliterated names). A strict 1.0 threshold would misclassify most Arabic text as "mixed." The 85% threshold was chosen empirically — it correctly classifies prompts like `"اشرح نظرية Einstein في الفيزياء"` as Arabic despite containing a Latin word.

### MCQ Detection Heuristics

Multiple-choice questions come in many formats in this corpus:

```
Format 1 (Arabic letters):     أ) ... ب) ... ج) ...
Format 2 (Latin letters):      A) ... B) ... C) ...
Format 3 (Numbered):           0) ... 1) ... 2) ...
Format 4 (Dash-separated):     أ - ... ب - ...
Format 5 (Keyword):            "اختر الإجابة الصحيحة"
```

The heuristic ORs multiple regex patterns. This intentionally over-detects (false positives are acceptable for profiling) rather than under-detecting, since the profile is informational.

---

## Step 2: Stratified Sampling — Why Reservoir Sampling?

### The Algorithm

We need representative samples from a 211K-row stream without knowing the total count in advance, and without loading everything into memory.

**Reservoir sampling** (Vitter, 1985) solves this:

```
For a reservoir of size k, processing the i-th item:
  if i < k:
      reservoir[i] = item
  else:
      j = random(0, i)
      if j < k:
          reservoir[j] = item
```

**Guarantee:** After processing all n items, each item has exactly k/n probability of being in the reservoir. This is uniform regardless of stream order.

### Why Stratified?

A single uniform sample of 50 rows from 211K would almost certainly miss rare categories. With 0.7% mixed-language rows (1,538 total), a uniform sample of 50 has only a ~30% chance of including even one mixed-language row.

By running separate reservoirs per stratum (language, question type, benchmark pattern), we guarantee coverage of every category. This is critical for quality auditing — we need to see the edge cases.

### Top-K Collectors

For length extremes (shortest/longest), we use a different structure: a sorted list that keeps the k most extreme items. This is not uniform sampling — it deterministically keeps the outliers.

---

## Step 3: Exact Dedup — Why Five Normalization Levels?

### The Hash Function

Each row is hashed with SHA-256:

```
hash(text) = SHA-256(utf-8-encode(text))
```

SHA-256 produces a 32-byte digest. For 211K rows, storing all hashes takes ~6.7 MB — trivially fits in memory. The collision probability for 211K items is approximately:

```
P(collision) ≈ n² / (2 · 2²⁵⁶) ≈ 10⁻⁶⁶
```

Effectively zero.

### The Five Levels

Each level applies progressively more aggressive normalization before hashing:

**L1: Raw pair** — `hash(user + "\0" + assistant)`

The null byte separator prevents `user="ab", assistant="cd"` from colliding with `user="a", assistant="bcd"`. L1 finds rows that are byte-for-byte identical.

**L2: Whitespace-normalized pair** — collapse `\s+` → single space, strip edges

Catches duplicates that differ only in trailing whitespace, extra newlines, or tab/space differences.

**L3: Think-stripped pair** — remove `<think>...</think>` blocks, then normalize whitespace

The `<think>` blocks are chain-of-thought reasoning traces. Two rows with identical user prompts and identical final answers but different reasoning paths are functionally identical for fine-tuning (the model learns from the final answer structure). We regex-strip:

```python
re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
```

**L4: User-only** — hash only the normalized user prompt

Two rows asking the same question but receiving different answers. We flag these but don't remove at this level (different answers may be valid).

**L5: User-only + Arabic canonicalization** — normalize Arabic orthographic variants

This is the Arabic-specific level. Arabic has multiple orthographic representations for the same word:

### Arabic Canonicalization Rules

| Transformation | Example | Why |
|---------------|---------|-----|
| Alef normalization: آ أ إ ٱ → ا | `أحمد` → `احمد` | These are interchangeable in informal Arabic text |
| Ya/Alef maksura: ى → ي | `على` → `علي` | Egyptian vs. standard spelling |
| Ta marbuta: ة → ه | `مدرسة` → `مدرسه` | Colloquial simplification |
| Strip tashkeel (diacritics) | `كِتَابٌ` → `كتاب` | Diacritics are optional in Arabic |
| Unify punctuation: ، → , and ؟ → ? | Arabic → Latin equivalents | Prevents punctuation-only mismatches |
| Remove invisible marks: RTL, LTR, ZWS, BOM | — | Unicode artifacts from copy-paste |
| Lowercase Latin characters | `Biology` → `biology` | For mixed-language prompts |

**Why not just jump to L5?** The layered approach reveals *where* the duplication comes from. If L1 and L2 show zero dupes but L3 shows 17, we know the duplication is specifically in the `<think>` traces. If L5 shows 176 but L4 shows only 22, the extra 154 are caused by Arabic normalization inconsistencies. This diagnostic information is valuable beyond just the final dedup count.

### Dedup Key Selection: Why L3?

We chose L3 (think-stripped pair) as the removal level because:

- It preserves rows that have the same question but different final answers (L4 would collapse those)
- It removes rows that are identical after stripping reasoning traces (these are true duplicates for fine-tuning)
- L5 is too aggressive — alef normalization can merge genuinely different questions in edge cases

---

## Step 4: Semantic Dedup — The Core Algorithm

### Why Embeddings + Cosine Similarity?

Exact hashing fails for semantic duplicates:

```
Row A:  "ما هي عاصمة مصر؟"
Row B:  "أخبرني عن عاصمة جمهورية مصر العربية"
```

These ask the same thing but share few n-grams. Embeddings map text to a dense vector space where semantic similarity corresponds to geometric proximity.

### Embedding Model: text-embedding-3-small

**Why this model?**

| Criterion | text-embedding-3-small | Alternatives |
|-----------|----------------------|--------------|
| Arabic quality | Strong (trained on multilingual data including Arabic) | multilingual-e5-base is good but lower quality |
| Dimensions | 1536 | 768 (e5-base), 3072 (3-large) |
| Token limit | 8,192 | Same for all OpenAI models |
| Cost per 211K rows | ~$0.08 | $0 (local) but 3-6 hrs on 2-core ARM |
| Latency | ~2 sec / batch of 200 | ~30 min total for local |

The 1536-dimension space provides good separation for near-duplicate detection. Higher dimensions (3-large, 3072) would give marginal quality improvement at 3x the cost and memory.

### What Gets Embedded?

**Only the user prompt**, after normalization:

```python
text = re.sub(r"<think>.*?</think>", "", text)  # strip think
text = re.sub(r"\s+", " ", text).strip()         # normalize whitespace
text = text[:5000]                               # truncate for token limit
```

**Why not embed both user and assistant?**

The assistant response is typically 3-5x longer than the user prompt and contains `<think>` traces that inflate the embedding with reasoning noise. Two rows with the same question but different `<think>` paths would have lower cosine similarity if we included assistant text, causing us to miss true duplicates.

The user prompt is the semantic anchor — if two prompts ask the same question, the rows are near-duplicates regardless of the answer.

### Token Limit Handling

OpenAI's `text-embedding-3-small` has an 8,192 token limit. Arabic tokenizes at ~1.5 characters per token with the cl100k_base tokenizer (compared to ~4 chars/token for English). This means:

```
8,192 tokens × 1.5 chars/token ≈ 12,288 characters
```

But the rate varies — dense Arabic with diacritics can tokenize as low as 1.0 char/token. We conservatively truncate at 5,000 characters (~3,300 tokens) to guarantee we never hit the limit, even for worst-case text.

For deduplication, truncation is safe: if two 10K-character prompts share the same first 5K characters, they're clearly near-duplicates.

### Similarity Search: Why FAISS IndexFlatIP?

**Problem:** Given 211K vectors of dimension 1536, find the k nearest neighbors for every vector by cosine similarity.

**Cosine similarity** between vectors \(a\) and \(b\):

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

If vectors are L2-normalized (‖a‖ = ‖b‖ = 1), this simplifies to:

```
cos(a, b) = a · b  (inner product)
```

We L2-normalize all vectors, then use FAISS `IndexFlatIP` (flat inner product index) for exact search.

**Why not an approximate index (IVF, HNSW)?**

| Index Type | Search Time | Accuracy | Build Time |
|-----------|-------------|----------|------------|
| **IndexFlatIP (exact)** | O(n · d) per query | 100% | O(n · d) |
| IndexIVFFlat | O(n/c · d) per query | ~95% with nprobe=10 | O(n · d + k-means) |
| IndexHNSW | O(d · log n) per query | ~97% | O(n · d · M) |

For n=211K, d=1536:
- Flat search: 211K × 1536 multiplications per query → ~324M FLOPs per query
- Total: 211K queries × 324M FLOPs = ~68 TFLOP
- On a 2-core ARM at ~10 GFLOPS: ~6,800 seconds (~113 minutes)

Approximate indexes would be faster (~10-20x) but at n=211K we're still in the "small enough for exact search" regime. The quality guarantee matters more than speed here — a missed near-duplicate is worse than waiting an extra 30 minutes.

**Memory layout:**

```
Embedding matrix:  211,058 × 1,536 × 4 bytes = 1.23 GB
FAISS index:       Same (flat index stores the vectors)
Score matrix:      211,058 × 11 × 4 bytes   = 8.9 MB  (top-10 + self)
Index matrix:      211,058 × 11 × 8 bytes   = 17.8 MB
Total:             ~2.5 GB
```

Fits in 8 GB RAM with headroom.

### Memory-Mapped Embeddings: Why Memmap?

The naive approach — accumulate embeddings in a Python list, then convert to numpy — fails:

```
Python float:    28 bytes each
NumPy float32:    4 bytes each
Overhead ratio:   7×
```

At 211K × 1536 dimensions:
- Python list of lists: 211K × 1536 × 28 = ~9.1 GB  (OOM on 8 GB machine)
- NumPy array:          211K × 1536 × 4  = 1.23 GB   (fits)

**Solution:** Write each batch of embeddings directly to a numpy memmap file on disk:

```python
mm = np.memmap("embeddings.npy", dtype=np.float32, mode="w+", shape=(n, d))
# Write batch directly
mm[start:end] = np.array(batch_vecs, dtype=np.float32)
mm.flush()
```

The OS pages data in/out of RAM as needed. Peak memory is one batch (~200 × 1536 × 4 = 1.2 MB) plus the memmap's working set.

### Threshold Calibration

The cosine similarity threshold \(τ\) determines what counts as a "near-duplicate." This is a precision-recall trade-off:

```
τ too high (0.99):  Only catches near-identical texts (high precision, low recall)
τ too low  (0.80):  Merges paraphrases and even different-topic texts (low precision, high recall)
```

We default to τ = 0.93 but emit sample pairs at four bands for manual calibration:

| Band | Similarity Range | Expected Content |
|------|-----------------|------------------|
| 0.99+ | Near-identical | Formatting differences, punctuation |
| 0.96–0.99 | Very similar | Minor rewording, synonym substitution |
| 0.93–0.96 | Similar | Same topic, different phrasing |
| 0.90–0.93 | Somewhat similar | Related but possibly distinct questions |

The user reviews 50 pairs from each band to decide where "duplicate" ends and "legitimately different" begins.

### Clustering: Union-Find

Given edges (pairs above threshold), we need connected components — groups of rows that are all transitively similar.

**Union-Find** (disjoint set) with path compression:

```python
parent = list(range(n))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
```

Time complexity: O(α(n)) per operation, where α is the inverse Ackermann function (effectively O(1)).

**Why not single-linkage clustering / DBSCAN?**

Union-Find on kNN edges *is* single-linkage clustering, just implemented efficiently. DBSCAN would add a minimum cluster size parameter we don't need — even a pair of duplicates should be flagged. And Union-Find is trivial to implement (15 lines) vs. pulling in sklearn.

### Representative Selection

Within each cluster, we keep one representative and remove the rest. The selection criterion:

```
representative = argmax_{i ∈ cluster} len(strip_think(assistant_text_i))
```

**Why longest non-think assistant response?**

- Longer answers tend to be more informative and complete
- Stripping `<think>` before measuring avoids biasing toward rows with verbose reasoning traces
- The user prompt is (approximately) the same across the cluster, so the differentiator is answer quality

---

## Step 5: Leakage Check — Pattern Matching

### What Is Benchmark Leakage?

If the training data contains questions *from* the evaluation benchmarks, the model's benchmark scores are inflated — it's "memorizing the test." This is the LLM equivalent of data leakage in ML.

### Detection Strategy

We use heuristic regex patterns, not exact matching against benchmark datasets (which we don't have locally). This means:

- **False positives are expected** — a question about biology triggers the MMLU pattern even if it's not from MMLU
- **False negatives are possible** — a benchmark question with unusual formatting might slip through
- **The goal is risk estimation**, not precise contamination measurement

### Signal Hierarchy

| Signal | Confidence | Count | Interpretation |
|--------|-----------|-------|---------------|
| Direct benchmark mention ("HELM", "benchmark", "leaderboard") | High | 182 | Likely meta-discussion about benchmarks, or leaked test preambles |
| Rubric leak ("الإجابة الصحيحة هي:") | Medium-High | 31,914 | Answer wrapper pattern common in benchmark-style Q&A |
| Benchmark keyword match | Low | 167,268 | Mostly false positives from topic overlap |

The 79.3% total flag rate demonstrates why heuristic matching alone can't drive removal decisions — you'd delete most of the dataset. Only high-confidence signals (182 + 31,914 rows) are used for actual removal in Step 6.

### Why Not N-Gram Overlap Against Benchmark Source?

The ideal leakage check would compute n-gram overlap between training prompts and actual benchmark questions. We don't do this because:

1. The benchmark source datasets aren't stored locally alongside this corpus
2. The pipeline is designed to be self-contained within `semantic-data-dedup/`
3. Pattern matching catches the structural signatures (MCQ formatting, answer wrappers) that are the strongest leakage indicators anyway

The architecture includes placeholder hooks for plugging in benchmark datasets later if available.

---

## Step 6: Final Assembly — Combining Removal Sets

### Set Operations

Each previous step produces a set of row indices to remove:

```
R_exact    = {rows removed by exact dedup (L3)}           # 17 rows
R_semantic = {rows removed by semantic dedup}              # 10,439 rows
R_leakage  = {high-confidence leakage flags}               # 200 rows (182 mentions + 18 rubric overlaps)

R_total = R_exact ∪ R_semantic ∪ R_leakage
```

A row is removed if it appears in *any* removal set. The union means we don't double-count — a row that's both an exact duplicate and a leakage flag is removed once.

### Removal Log

Every removed row is logged with its reason(s):

```json
{"row_index": 12345, "reasons": ["semantic_duplicate", "leakage_high_confidence"]}
```

This enables post-hoc analysis: if the clean dataset performs unexpectedly on a benchmark, we can check whether we over-removed in that category.

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dedup order | Hash → Semantic → Leakage | Cheapest first; each step reduces input for the next |
| Hash algorithm | SHA-256 | Zero collision risk at 211K scale; fast |
| Dedup level for removal | L3 (think-stripped pair) | Removes true duplicates without over-merging |
| Embedding model | text-embedding-3-small | Best Arabic quality per dollar; existing API key |
| Embedding surface | User prompt only | Avoids `<think>` noise; prompts are the semantic anchor |
| Truncation limit | 5,000 chars | Arabic tokenizes at ~1.5 chars/token; safe margin under 8,192 token limit |
| Similarity metric | Cosine (via L2-normalized inner product) | Scale-invariant; standard for text embeddings |
| Search algorithm | FAISS IndexFlatIP (exact) | 211K is small enough for exact search; no recall loss |
| Threshold | 0.93 (with calibration samples) | Balance between catching paraphrases and preserving distinct questions |
| Clustering | Union-Find (connected components) | O(n·α(n)); simpler and faster than DBSCAN/hierarchical |
| Representative selection | Longest non-think assistant response | Prefers more complete answers |
| Memory strategy | Numpy memmap for embeddings | Avoids 7× Python float overhead; fits in 8 GB |
| Leakage detection | Regex heuristics | No benchmark source data available; patterns catch structural signatures |
| Leakage removal | High-confidence only | Keyword matching has >50% false positive rate |

---

## Complexity Analysis

| Step | Time | Memory | I/O |
|------|------|--------|-----|
| 01_profile | O(n) | O(1) | 1 pass over 616 MB |
| 02_sample | O(n) | O(k · s) where s = number of strata | 1 pass |
| 03_exact_dedup | O(n) | O(n) for hash sets (~6.7 MB) | 2 passes (hash + write) |
| 04_semantic_dedup | O(n · b) embed + O(n² · d) search | O(n · d) for FAISS (~2.5 GB) | API calls + memmap |
| 05_leakage_check | O(n · p) where p = number of patterns | O(1) | 1 pass |
| 06_build_clean | O(n) | O(\|R\|) for removal sets | 1 pass |

The bottleneck is Step 4's FAISS search at O(n² · d). For n=211K and d=1536, this is ~68 TFLOP — roughly 30-60 minutes on a 2-core ARM machine.

---

## Arabic-Specific Considerations

### Why Arabic Dedup Is Harder Than English

1. **Orthographic ambiguity:** The same Arabic word can be written 4+ ways depending on alef form, diacritics, and ta-marbuta. English has no equivalent (aside from British/American spelling).

2. **Tokenizer inefficiency:** The cl100k_base tokenizer was primarily trained on English. Arabic characters consume 1.5-3× more tokens per character than English, which means:
   - Shorter effective context window for embeddings
   - Higher API costs per character
   - Need for more aggressive truncation

3. **Bidirectional text:** Arabic is RTL with frequent LTR insertions (numbers, Latin terms). Unicode control characters (RTL mark U+200F, LTR mark U+200E) are invisible but affect string comparison. Our canonicalization strips these.

4. **Dialect variation:** The corpus contains MSA (Modern Standard Arabic) and some dialectal text. Two prompts asking the same question in MSA vs. Egyptian Arabic are semantic duplicates but have low lexical overlap. Embeddings handle this well; hashing does not.

5. **Diacritics as optional metadata:** Arabic diacritics (tashkeel) are pronunciation guides, not semantic content. `كِتَابٌ` and `كتاب` are the same word. Our L5 normalization strips these before hashing, and the embedding model largely ignores them.

---

## Final Results

| Stage | Input | Output | Removed | % |
|-------|-------|--------|---------|---|
| Source | — | 211,075 | — | — |
| Exact dedup (L3) | 211,075 | 211,058 | 17 | 0.01% |
| Semantic dedup (τ=0.93) | 211,058 | 200,619 | 10,439 | 4.95% |
| Leakage removal (high-confidence) | — | — | 200 | 0.09% |
| **Final clean** | **211,075** | **200,428** | **10,647** | **5.04%** |

The semantic dedup step accounts for 98% of all removals. The dataset was already very clean at the exact-duplicate level (only 17 rows), which validates that the original data preparation was thorough — the remaining duplicates were paraphrases and reworded questions that only embedding-based methods could catch.
