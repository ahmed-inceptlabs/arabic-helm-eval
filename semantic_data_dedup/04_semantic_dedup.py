#!/usr/bin/env python3
"""Semantic near-duplicate detection using embeddings and FAISS.

Memory-efficient: uses numpy memmap for embeddings so the full 211K x 1536
matrix never lives in a Python list. Rows are streamed from disk on demand.

Supports two embedding backends:
  --backend openai   (default) Uses text-embedding-3-small via the OpenAI API
  --backend local    Uses intfloat/multilingual-e5-base via sentence-transformers

Usage:
    python 04_semantic_dedup.py --backend openai
    python 04_semantic_dedup.py --backend openai --threshold 0.95
    python 04_semantic_dedup.py --backend local
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "data" / "03_deduped.jsonl"
REPORT_DIR = SCRIPT_DIR / "reports"
DATA_DIR = SCRIPT_DIR / "data"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
WHITESPACE = re.compile(r"\s+")

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536
LOCAL_MODEL = "intfloat/multilingual-e5-base"
LOCAL_DIM = 768
OPENAI_BATCH_SIZE = 200
LOCAL_BATCH_SIZE = 64
MAX_EMBED_CHARS = 5000

SIMILARITY_BANDS = [
    ("0.99+", 0.99, 1.01),
    ("0.96-0.99", 0.96, 0.99),
    ("0.93-0.96", 0.93, 0.96),
    ("0.90-0.93", 0.90, 0.93),
]
BAND_SAMPLE_SIZE = 50


def normalize_prompt(text: str) -> str:
    text = THINK_PATTERN.sub("", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def extract_user_text(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def load_texts_only():
    """Load only normalized user prompts (not full rows) to save memory."""
    texts = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(normalize_prompt(extract_user_text(row)))
    return texts


def count_rows():
    n = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_api_key() -> str:
    cred_path = SCRIPT_DIR.parent / "credentials.conf"
    if cred_path.exists():
        with open(cred_path) as f:
            for line in f:
                if line.strip().startswith("openaiApiKey:"):
                    key = line.split(":", 1)[1].strip().strip('"').strip("'")
                    if key and key != "lm-studio":
                        return key
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    raise RuntimeError(
        "No OpenAI API key found. Set OPENAI_API_KEY or add to ../credentials.conf"
    )


def _truncate_text(text: str) -> str:
    if len(text) > MAX_EMBED_CHARS:
        return text[:MAX_EMBED_CHARS]
    return text


def _send_batch(client, batch: list[str], label: str, depth: int = 0) -> list:
    """Send a single batch with retries. Splits on token-limit errors."""
    batch = [_truncate_text(t) for t in batch]
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
            return [item.embedding for item in resp.data]
        except Exception as e:
            err_str = str(e)
            is_token_error = any(k in err_str for k in (
                "max_tokens", "maximum context length",
                "maximum request size", "maximum input length",
                "tokens per request",
            ))
            if is_token_error:
                if len(batch) > 1:
                    mid = len(batch) // 2
                    if depth < 3:
                        print(f"  {label}: splitting batch of {len(batch)}", file=sys.stderr)
                    left = _send_batch(client, batch[:mid], f"{label}L", depth + 1)
                    right = _send_batch(client, batch[mid:], f"{label}R", depth + 1)
                    return left + right
                else:
                    halved = batch[0][:len(batch[0]) // 2]
                    print(f"  {label}: single text too long ({len(batch[0])} chars), halving", file=sys.stderr)
                    batch = [halved]
                    continue
            if "rate" in err_str.lower() or "429" in err_str:
                wait = min(2 ** (attempt + 2), 60)
            else:
                wait = 2 ** attempt
            print(f"  {label}: API error (attempt {attempt+1}): {e}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed {label} after retries")


def _find_memmap_progress(path: Path, total: int, dim: int) -> int:
    """Find how many rows are already embedded in a memmap file."""
    expected_bytes = total * dim * 4
    if not path.exists() or path.stat().st_size < expected_bytes:
        return 0
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dim))
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        if np.any(mm[mid] != 0):
            lo = mid + 1
        else:
            hi = mid
    del mm
    return lo


def embed_openai_to_memmap(texts: list[str], out_path: Path) -> np.ndarray:
    """Embed texts and write directly to a memory-mapped numpy file.

    Resumes from existing memmap or checkpoint, writing new batches
    directly to the memmap without accumulating in Python memory.
    """
    from openai import OpenAI

    api_key = read_api_key()
    client = OpenAI(api_key=api_key)

    total = len(texts)
    dim = OPENAI_DIM
    ckpt_path = DATA_DIR / "04_embeddings_checkpoint.npy"
    resume_from = 0

    memmap_progress = _find_memmap_progress(out_path, total, dim)
    if memmap_progress >= total:
        print(f"  Full embeddings already in memmap ({total:,} x {dim})", file=sys.stderr)
        return np.memmap(out_path, dtype=np.float32, mode="r", shape=(total, dim))

    if memmap_progress > 0:
        resume_from = memmap_progress
        print(f"  Resuming from memmap progress: {resume_from:,}/{total:,}", file=sys.stderr)
        mm = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(total, dim))
    else:
        mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(total, dim))
        if ckpt_path.exists():
            cached = np.load(ckpt_path)
            resume_from = min(cached.shape[0], total)
            mm[:resume_from] = cached[:resume_from]
            mm.flush()
            del cached
            print(f"  Loaded {resume_from:,} rows from checkpoint into memmap", file=sys.stderr)

    for start in range(resume_from, total, OPENAI_BATCH_SIZE):
        batch = texts[start : start + OPENAI_BATCH_SIZE]
        vecs = _send_batch(client, batch, f"batch@{start}")
        end = start + len(vecs)
        mm[start:end] = np.array(vecs, dtype=np.float32)

        if end % 5000 < OPENAI_BATCH_SIZE:
            mm.flush()

        done = min(start + OPENAI_BATCH_SIZE, total)
        print(f"  Embedded {done:,}/{total:,}", file=sys.stderr)

    mm.flush()
    del mm

    return np.memmap(out_path, dtype=np.float32, mode="r", shape=(total, dim))


def embed_local(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model {LOCAL_MODEL}...", file=sys.stderr)
    model = SentenceTransformer(LOCAL_MODEL)

    prefixed = [f"query: {t}" for t in texts]

    print(f"  Encoding {len(texts):,} texts (batch_size={LOCAL_BATCH_SIZE})...", file=sys.stderr)
    embeddings = model.encode(
        prefixed,
        batch_size=LOCAL_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def normalize_l2(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def normalize_l2_memmap(src_path: Path, total: int, dim: int) -> np.ndarray:
    """L2-normalize a memmap file in-place in chunks to avoid loading it all."""
    mm = np.memmap(src_path, dtype=np.float32, mode="r+", shape=(total, dim))
    chunk = 10000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        block = np.array(mm[start:end])
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mm[start:end] = block / norms
    mm.flush()
    del mm
    return np.memmap(src_path, dtype=np.float32, mode="r", shape=(total, dim))


def find_neighbors(embeddings: np.ndarray, k: int = 10):
    """Use FAISS to find top-k nearest neighbors by cosine similarity."""
    import faiss

    n, d = embeddings.shape
    print(f"  Building FAISS index ({n:,} x {d})...", file=sys.stderr)

    emb_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(d)
    index.add(emb_contiguous)

    print(f"  Searching top-{k} neighbors...", file=sys.stderr)

    batch_size = 1000
    all_scores = []
    all_indices = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        query = np.ascontiguousarray(emb_contiguous[start:end])
        scores, indices = index.search(query, k + 1)
        all_scores.append(scores)
        all_indices.append(indices)
        if end % 50000 == 0 or end == n:
            print(f"    Searched {end:,}/{n:,}", file=sys.stderr)

    return np.vstack(all_scores), np.vstack(all_indices)


def build_similarity_histogram(scores: np.ndarray, indices: np.ndarray) -> dict:
    n = scores.shape[0]
    max_sims = np.full(n, -1.0)

    for i in range(n):
        for j_idx in range(scores.shape[1]):
            if indices[i, j_idx] != i:
                max_sims[i] = float(scores[i, j_idx])
                break

    bins = [0.0, 0.5, 0.7, 0.8, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01]
    counts, _ = np.histogram(max_sims, bins=bins)

    histogram = {}
    for i in range(len(bins) - 1):
        label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        histogram[label] = int(counts[i])

    return {
        "histogram": histogram,
        "stats": {
            "mean": round(float(max_sims.mean()), 4),
            "median": round(float(np.median(max_sims)), 4),
            "p90": round(float(np.percentile(max_sims, 90)), 4),
            "p95": round(float(np.percentile(max_sims, 95)), 4),
            "p99": round(float(np.percentile(max_sims, 99)), 4),
            "max": round(float(max_sims.max()), 4),
        },
    }


def collect_band_samples(
    scores: np.ndarray,
    indices: np.ndarray,
    texts: list[str],
) -> dict:
    """Collect sample pairs at different similarity bands for threshold calibration."""
    band_pairs = {label: [] for label, _, _ in SIMILARITY_BANDS}
    n = scores.shape[0]

    for i in range(n):
        for j_idx in range(scores.shape[1]):
            j = int(indices[i, j_idx])
            if j <= i:
                continue
            sim = float(scores[i, j_idx])

            for label, lo, hi in SIMILARITY_BANDS:
                if lo <= sim < hi and len(band_pairs[label]) < BAND_SAMPLE_SIZE:
                    band_pairs[label].append({
                        "similarity": round(sim, 4),
                        "row_i": i,
                        "row_j": j,
                        "user_i": texts[i][:500],
                        "user_j": texts[j][:500],
                    })
                    break

    return band_pairs


def cluster_duplicates(
    scores: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    n: int,
) -> list[list[int]]:
    """Connected-component clustering on edges above threshold."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_count = 0
    for i in range(n):
        for j_idx in range(scores.shape[1]):
            j = int(indices[i, j_idx])
            if j == i:
                continue
            if float(scores[i, j_idx]) >= threshold:
                union(i, j)
                edge_count += 1

    print(f"  Edges above threshold {threshold}: {edge_count:,}", file=sys.stderr)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = [v for v in groups.values() if len(v) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters


def pick_representatives_streaming(clusters: list[list[int]]) -> dict:
    """Stream through the JSONL to pick best representative per cluster.

    Returns {row_index: True} for rows to REMOVE (all non-representatives).
    """
    cluster_members = {}
    for cluster in clusters:
        for idx in cluster:
            cluster_members[idx] = cluster

    best = {}
    best_len = {}
    for cluster in clusters:
        cid = id(cluster)
        best[cid] = cluster[0]
        best_len[cid] = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in cluster_members:
                continue
            cluster = cluster_members[idx]
            cid = id(cluster)
            try:
                row = json.loads(line.strip())
                for m in row.get("messages", []):
                    if m.get("role") == "assistant":
                        stripped = THINK_PATTERN.sub("", m.get("content", ""))
                        slen = len(stripped.strip())
                        if slen > best_len[cid]:
                            best_len[cid] = slen
                            best[cid] = idx
            except json.JSONDecodeError:
                pass

    remove_set = set()
    for cluster in clusters:
        cid = id(cluster)
        rep = best[cid]
        for idx in cluster:
            if idx != rep:
                remove_set.add(idx)

    return remove_set, best


def main():
    parser = argparse.ArgumentParser(description="Semantic near-dedup")
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--threshold", type=float, default=0.93,
                        help="Cosine similarity threshold for dedup (default: 0.93)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of nearest neighbors to retrieve (default: 10)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding, load cached embeddings from data/04_embeddings.npy")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "04_threshold_samples").mkdir(parents=True, exist_ok=True)

    print("Loading texts from deduped file...", file=sys.stderr)
    texts = load_texts_only()
    total = len(texts)
    print(f"  Loaded {total:,} texts ({sys.getsizeof(texts)/1024/1024:.0f} MB list overhead)", file=sys.stderr)

    embed_cache = DATA_DIR / "04_embeddings.npy"
    dim = OPENAI_DIM if args.backend == "openai" else LOCAL_DIM

    if args.skip_embed and embed_cache.exists():
        print("Loading cached embeddings...", file=sys.stderr)
        embeddings = np.memmap(embed_cache, dtype=np.float32, mode="r", shape=(total, dim))
    else:
        print(f"Embedding with backend={args.backend}...", file=sys.stderr)
        if args.backend == "openai":
            embeddings = embed_openai_to_memmap(texts, embed_cache)
            print("  Normalizing L2...", file=sys.stderr)
            embeddings = normalize_l2_memmap(embed_cache, total, dim)
        else:
            embeddings = embed_local(texts)
            np.save(embed_cache, embeddings)

        print(f"  Embeddings saved to {embed_cache}", file=sys.stderr)

    print(f"Embeddings shape: {embeddings.shape}", file=sys.stderr)

    scores, indices = find_neighbors(embeddings, k=args.top_k)

    print("Building similarity histogram...", file=sys.stderr)
    histogram = build_similarity_histogram(scores, indices)
    hist_path = REPORT_DIR / "04_similarity_histogram.json"
    with open(hist_path, "w") as f:
        json.dump(histogram, f, indent=2)
    print(f"  Histogram saved to {hist_path}", file=sys.stderr)

    print("Collecting threshold calibration samples...", file=sys.stderr)
    band_samples = collect_band_samples(scores, indices, texts)
    for label, pairs in band_samples.items():
        out_path = REPORT_DIR / "04_threshold_samples" / f"band_{label.replace('+','plus')}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Band {label}: {len(pairs)} sample pairs", file=sys.stderr)

    print(f"\nClustering at threshold={args.threshold}...", file=sys.stderr)
    clusters = cluster_duplicates(scores, indices, args.threshold, total)

    total_in_clusters = sum(len(c) for c in clusters)
    removable = total_in_clusters - len(clusters)

    print(f"  Clusters found: {len(clusters):,}", file=sys.stderr)
    print(f"  Rows in clusters: {total_in_clusters:,}", file=sys.stderr)
    print(f"  Removable: {removable:,}", file=sys.stderr)

    print("Picking representatives (streaming)...", file=sys.stderr)
    remove_set, best_map = pick_representatives_streaming(clusters)

    cluster_data = []
    for ci, cluster in enumerate(clusters):
        cid = id(cluster)
        rep = best_map[cid]
        to_remove = [i for i in cluster if i != rep]
        cluster_data.append({
            "cluster_id": ci,
            "size": len(cluster),
            "representative": rep,
            "removed": to_remove,
            "representative_preview": texts[rep][:300],
        })

    cluster_path = DATA_DIR / "04_semantic_clusters.json"
    with open(cluster_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)

    print("Writing deduplicated output (streaming)...", file=sys.stderr)
    deduped_path = DATA_DIR / "04_deduped.jsonl"
    kept = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(deduped_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if idx not in remove_set:
                fout.write(line + "\n")
                kept += 1

    removed = total - kept
    print(f"\nSemantic dedup output: {deduped_path}", file=sys.stderr)
    print(f"  Kept: {kept:,}, Removed: {removed:,}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("SEMANTIC DEDUP SUMMARY")
    print("=" * 60)
    print(f"Input rows:      {total:,}")
    print(f"Backend:         {args.backend}")
    print(f"Threshold:       {args.threshold}")
    print(f"Clusters:        {len(clusters):,}")
    print(f"Rows in clusters:{total_in_clusters:,}")
    print(f"Removable:       {removable:,} ({100*removable/max(total,1):.1f}%)")
    print(f"Kept:            {kept:,}")
    print(f"\nSimilarity stats: {histogram['stats']}")
    print(f"\nHistogram:")
    for band, count in histogram["histogram"].items():
        bar = "#" * min(count // 500, 50)
        print(f"  {band}: {count:>8,}  {bar}")


if __name__ == "__main__":
    main()
