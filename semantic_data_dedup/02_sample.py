#!/usr/bin/env python3
"""Stratified sampling for manual quality audit.

Streams the JSONL and collects small representative subsets using reservoir
sampling, grouped by language, question type, length, and benchmark pattern.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "ar93_en7_mcq85_open15_cot96_211k.jsonl"
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "reports" / "02_samples"

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Broader MCQ detection: includes numbered options, lettered options, and keyword patterns
MCQ_PATTERNS = [
    re.compile(r"\n\s*[أبجدهـ][\)\.]\s"),
    re.compile(r"\n\s*[A-E][\)\.]\s"),
    re.compile(r"\n\s*\d+[\)\.]\s.*\n\s*\d+[\)\.]\s"),
    re.compile(r"اختر\s*(الإجابة|رقم|حرف|الصحيح)", re.IGNORECASE),
    re.compile(r"Choose the correct", re.IGNORECASE),
    re.compile(r"متعدد\s*الاختيار"),
    re.compile(r"multiple.choice", re.IGNORECASE),
    re.compile(r"\bأ\)\s.*\bب\)\s", re.DOTALL),
    re.compile(r"\bA\)\s.*\bB\)\s", re.DOTALL),
    re.compile(r"\n\s*أ\s*[\-–—:]\s"),
    re.compile(r"\n\s*[أا]ختر\b"),
]

BENCHMARK_PATTERNS = {
    "aratrust_safety": re.compile(
        r"إهانة|عنصري|تحرش|خصوصية|privacy|bias|toxic|offensive|إساءة|تمييز",
        re.IGNORECASE,
    ),
    "mmlu_subject": re.compile(
        r"anatomy|biology|chemistry|physics|mathematics|algebra|history|psychology|"
        r"law|medicine|economics|sociology|philosophy|computer science|astronomy",
        re.IGNORECASE,
    ),
    "alghafa_sentiment": re.compile(
        r"sentiment|إيجابي|سلبي|رأي|positive|negative|مراجعة|review|تقييم",
        re.IGNORECASE,
    ),
    "exam_style": re.compile(
        r"امتحان|اختبار|سؤال.*اختيار|exam|test question|مقرر|منهج|curriculum",
        re.IGNORECASE,
    ),
    "alrage_generation": re.compile(
        r"اشرح|وضح|اكتب مقال|explain in detail|write an essay|لخص|summarize",
        re.IGNORECASE,
    ),
}


def classify_language(text: str) -> str:
    ar = len(ARABIC_RANGE.findall(text))
    la = len(LATIN_RANGE.findall(text))
    total = ar + la
    if total == 0:
        return "other"
    frac = ar / total
    if frac >= 0.85:
        return "arabic"
    if frac <= 0.15:
        return "english"
    return "mixed"


def is_mcq(text: str) -> bool:
    return any(p.search(text) for p in MCQ_PATTERNS)


class ReservoirSampler:
    """Reservoir sampling: keeps exactly k items uniformly at random from a stream."""

    def __init__(self, k: int):
        self.k = k
        self.reservoir = []
        self.n = 0

    def add(self, item):
        self.n += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.reservoir[j] = item

    def items(self):
        return self.reservoir


class TopKCollector:
    """Keep top-k items by a score (highest or lowest)."""

    def __init__(self, k: int, highest: bool = True):
        self.k = k
        self.highest = highest
        self.items_list = []

    def add(self, item, score):
        self.items_list.append((score, item))
        if len(self.items_list) > self.k * 3:
            self._prune()

    def _prune(self):
        self.items_list.sort(key=lambda x: x[0], reverse=self.highest)
        self.items_list = self.items_list[: self.k]

    def items(self):
        self._prune()
        return [item for _, item in self.items_list]


def build_sample_row(idx: int, row: dict) -> dict:
    """Wrap a row with its index for traceability."""
    return {"row_index": idx, **row}


def sample(input_file=None, sample_dir=None, sample_size=None, seed=None):
    input_file = Path(input_file) if input_file else DEFAULT_INPUT
    sample_dir = Path(sample_dir) if sample_dir else DEFAULT_SAMPLE_DIR
    k = sample_size or DEFAULT_SAMPLE_SIZE

    random.seed(seed or DEFAULT_SEED)
    SAMPLE_DIR = sample_dir
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    uniform = ReservoirSampler(k)
    by_lang = defaultdict(lambda: ReservoirSampler(k))
    by_type = defaultdict(lambda: ReservoirSampler(k))
    by_benchmark = defaultdict(lambda: ReservoirSampler(k))

    longest_think = TopKCollector(k, highest=True)
    shortest_user = TopKCollector(k, highest=False)
    longest_user = TopKCollector(k, highest=True)
    shortest_asst = TopKCollector(k, highest=False)

    INPUT_FILE = input_file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            msgs = row.get("messages", [])
            user_text = ""
            asst_text = ""
            for m in msgs:
                if m["role"] == "user":
                    user_text = m.get("content", "")
                elif m["role"] == "assistant":
                    asst_text = m.get("content", "")

            sample_row = build_sample_row(idx, row)

            uniform.add(sample_row)

            lang = classify_language(user_text)
            by_lang[lang].add(sample_row)

            qtype = "mcq" if is_mcq(user_text) else "open_ended"
            by_type[qtype].add(sample_row)

            for bname, pattern in BENCHMARK_PATTERNS.items():
                if pattern.search(user_text):
                    by_benchmark[bname].add(sample_row)

            think_matches = THINK_PATTERN.findall(asst_text)
            if think_matches:
                total_think_len = sum(len(t) for t in think_matches)
                longest_think.add(sample_row, total_think_len)

            shortest_user.add(sample_row, len(user_text))
            longest_user.add(sample_row, len(user_text))
            shortest_asst.add(sample_row, len(asst_text))

            if idx % 50000 == 0 and idx > 0:
                print(f"  ... sampled through {idx:,} rows", file=sys.stderr)

    slices = {
        "uniform_random": uniform.items(),
        **{f"lang_{k}": v.items() for k, v in by_lang.items()},
        **{f"type_{k}": v.items() for k, v in by_type.items()},
        **{f"benchmark_{k}": v.items() for k, v in by_benchmark.items()},
        "longest_think": longest_think.items(),
        "shortest_user_prompt": shortest_user.items(),
        "longest_user_prompt": longest_user.items(),
        "shortest_assistant": shortest_asst.items(),
    }

    index = {}
    for name, items in slices.items():
        out_path = SAMPLE_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        index[name] = {
            "file": str(out_path.relative_to(SCRIPT_DIR)),
            "count": len(items),
            "row_indices": sorted(item["row_index"] for item in items),
        }
        print(f"  {name}: {len(items)} samples")

    index_path = SAMPLE_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nAll samples saved to: {SAMPLE_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified reservoir sampling for manual quality audit")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to input JSONL file (default: %(default)s)")
    parser.add_argument("--output-dir", default=str(DEFAULT_SAMPLE_DIR),
                        help="Directory for sample outputs (default: %(default)s)")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Rows per sample slice (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s)")
    args = parser.parse_args()
    sample(input_file=args.input, sample_dir=args.output_dir,
           sample_size=args.sample_size, seed=args.seed)
