#!/usr/bin/env python3
"""Streaming profiler for the Arabic fine-tuning JSONL corpus.

Reads the file line-by-line (constant memory) and emits a JSON report with:
- Row counts and parse failures
- Message structure distribution
- Text length distributions (user / assistant)
- Language mix (Arabic vs English vs mixed)
- <think> tag prevalence and lengths
- MCQ vs open-ended classification
- Arabic normalization variant stats
- Boilerplate / answer-wrapper frequency
"""

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "ar93_en7_mcq85_open15_cot96_211k.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

MCQ_MARKERS_AR = re.compile(r"[أبجدهـ]\)")
MCQ_MARKERS_LATIN = re.compile(r"\b[A-E]\)")
MCQ_NEWLINE_OPTIONS = re.compile(r"\n\s*[أبجدهـA-E][\)\.]\s")
MCQ_CHOICE_KEYWORDS = re.compile(r"اختر|اختيار|Choose|choose|multiple.choice|متعدد", re.IGNORECASE)

ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آ أ إ ٱ
TA_MARBUTA = re.compile(r"\u0629")  # ة
YA_VARIANTS = re.compile(r"\u0649")  # ى (alef maksura)
TASHKEEL = re.compile(r"[\u064B-\u065F\u0670]")

ANSWER_WRAPPERS = [
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب الصحيح هو:",
    "الجواب هو:",
    "The correct answer is:",
    "The answer is:",
]


def classify_language(text: str) -> str:
    """Classify text as arabic, english, or mixed based on script character counts."""
    ar_count = len(ARABIC_RANGE.findall(text))
    lat_count = len(LATIN_RANGE.findall(text))
    total = ar_count + lat_count
    if total == 0:
        return "other"
    ar_frac = ar_count / total
    if ar_frac >= 0.85:
        return "arabic"
    if ar_frac <= 0.15:
        return "english"
    return "mixed"


def is_mcq(user_text: str) -> bool:
    """Heuristic: does the user prompt look like a multiple-choice question?"""
    if MCQ_NEWLINE_OPTIONS.search(user_text):
        return True
    markers = len(MCQ_MARKERS_AR.findall(user_text)) + len(MCQ_MARKERS_LATIN.findall(user_text))
    if markers >= 2:
        return True
    if MCQ_CHOICE_KEYWORDS.search(user_text) and markers >= 1:
        return True
    return False


class StreamingStats:
    """Accumulate min/max/mean/stddev/percentiles without storing all values."""

    def __init__(self):
        self.n = 0
        self.total = 0
        self.sq_total = 0.0
        self.mn = float("inf")
        self.mx = float("-inf")
        self.buckets = Counter()

    def add(self, v):
        self.n += 1
        self.total += v
        self.sq_total += v * v
        if v < self.mn:
            self.mn = v
        if v > self.mx:
            self.mx = v
        bucket = int(v // 100) * 100
        self.buckets[bucket] += 1

    def to_dict(self):
        if self.n == 0:
            return {"count": 0}
        mean = self.total / self.n
        var = max(0, self.sq_total / self.n - mean * mean)
        return {
            "count": self.n,
            "mean": round(mean, 1),
            "std": round(math.sqrt(var), 1),
            "min": self.mn,
            "max": self.mx,
            "total": self.total,
            "histogram_100": dict(sorted(self.buckets.items())[:30]),
        }


def profile(input_file=None, report_dir=None):
    input_file = Path(input_file) if input_file else DEFAULT_INPUT
    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR

    INPUT_FILE = input_file
    REPORT_DIR = report_dir
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    parse_failures = 0
    msg_count_dist = Counter()
    role_patterns = Counter()

    user_len_stats = StreamingStats()
    asst_len_stats = StreamingStats()

    lang_user = Counter()
    lang_asst = Counter()

    think_count = 0
    think_len_stats = StreamingStats()
    no_think_count = 0

    mcq_count = 0
    open_count = 0

    alef_variant_rows = 0
    ta_marbuta_rows = 0
    ya_variant_rows = 0
    tashkeel_rows = 0

    wrapper_counts = Counter()
    empty_user = 0
    empty_asst = 0

    code_block_rows = 0
    url_rows = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_failures += 1
                continue

            msgs = row.get("messages", [])
            msg_count_dist[len(msgs)] += 1
            roles = "->".join(m.get("role", "?") for m in msgs)
            role_patterns[roles] += 1

            user_text = ""
            asst_text = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                elif m.get("role") == "assistant":
                    asst_text = m.get("content", "")

            u_len = len(user_text)
            a_len = len(asst_text)
            user_len_stats.add(u_len)
            asst_len_stats.add(a_len)

            if u_len == 0:
                empty_user += 1
            if a_len == 0:
                empty_asst += 1

            lang_user[classify_language(user_text)] += 1
            lang_asst[classify_language(asst_text)] += 1

            think_matches = THINK_PATTERN.findall(asst_text)
            if think_matches:
                think_count += 1
                for tm in think_matches:
                    think_len_stats.add(len(tm))
            else:
                no_think_count += 1

            if is_mcq(user_text):
                mcq_count += 1
            else:
                open_count += 1

            combined = user_text + asst_text
            if ALEF_VARIANTS.search(combined):
                alef_variant_rows += 1
            if TA_MARBUTA.search(combined):
                ta_marbuta_rows += 1
            if YA_VARIANTS.search(combined):
                ya_variant_rows += 1
            if TASHKEEL.search(combined):
                tashkeel_rows += 1

            for wrapper in ANSWER_WRAPPERS:
                if wrapper in asst_text:
                    wrapper_counts[wrapper] += 1

            if "```" in combined:
                code_block_rows += 1
            if re.search(r"https?://", combined):
                url_rows += 1

            if total_rows % 50000 == 0:
                print(f"  ... processed {total_rows:,} rows", file=sys.stderr)

    report = {
        "total_rows": total_rows,
        "parse_failures": parse_failures,
        "message_count_distribution": dict(msg_count_dist),
        "role_patterns": dict(role_patterns.most_common(20)),
        "user_length": user_len_stats.to_dict(),
        "assistant_length": asst_len_stats.to_dict(),
        "empty_user_content": empty_user,
        "empty_assistant_content": empty_asst,
        "language_user": dict(lang_user),
        "language_assistant": dict(lang_asst),
        "think_tag": {
            "rows_with_think": think_count,
            "rows_without_think": no_think_count,
            "think_fraction": round(think_count / max(total_rows, 1), 4),
            "think_block_length": think_len_stats.to_dict(),
        },
        "question_type": {
            "mcq": mcq_count,
            "open_ended": open_count,
            "mcq_fraction": round(mcq_count / max(total_rows, 1), 4),
        },
        "arabic_normalization_stats": {
            "rows_with_alef_variants": alef_variant_rows,
            "rows_with_ta_marbuta": ta_marbuta_rows,
            "rows_with_ya_variants": ya_variant_rows,
            "rows_with_tashkeel": tashkeel_rows,
        },
        "answer_wrapper_counts": dict(wrapper_counts.most_common()),
        "code_block_rows": code_block_rows,
        "url_rows": url_rows,
        "filename_metadata_validation": {
            "expected_ar_pct": 93,
            "actual_ar_user_pct": round(
                100 * lang_user.get("arabic", 0) / max(total_rows, 1), 1
            ),
            "expected_mcq_pct": 85,
            "actual_mcq_pct": round(100 * mcq_count / max(total_rows, 1), 1),
            "expected_cot_pct": 96,
            "actual_think_pct": round(
                100 * think_count / max(total_rows, 1), 1
            ),
            "expected_rows": "211k",
            "actual_rows": total_rows,
        },
    }

    report_path = REPORT_DIR / "01_profile_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("DATASET PROFILE SUMMARY")
    print("=" * 70)
    print(f"Total rows:          {total_rows:,}")
    print(f"Parse failures:      {parse_failures}")
    print(f"Empty user content:  {empty_user}")
    print(f"Empty asst content:  {empty_asst}")
    print(f"\nMessage structure:   {dict(msg_count_dist)}")
    print(f"Role patterns (top): {dict(role_patterns.most_common(5))}")
    print(f"\nUser text length:    mean={user_len_stats.to_dict()['mean']}, "
          f"min={user_len_stats.mn}, max={user_len_stats.mx}")
    print(f"Asst text length:    mean={asst_len_stats.to_dict()['mean']}, "
          f"min={asst_len_stats.mn}, max={asst_len_stats.mx}")
    print(f"\nLanguage (user):     {dict(lang_user)}")
    print(f"Language (asst):     {dict(lang_asst)}")
    print(f"\n<think> prevalence:  {think_count:,} / {total_rows:,} "
          f"({100*think_count/max(total_rows,1):.1f}%)")
    print(f"MCQ detected:        {mcq_count:,} / {total_rows:,} "
          f"({100*mcq_count/max(total_rows,1):.1f}%)")
    print(f"Open-ended:          {open_count:,} / {total_rows:,} "
          f"({100*open_count/max(total_rows,1):.1f}%)")
    print(f"\nAnswer wrappers:     {dict(wrapper_counts.most_common(5))}")
    print(f"Code block rows:     {code_block_rows:,}")
    print(f"URL rows:            {url_rows:,}")
    print(f"\n--- Filename Metadata Validation ---")
    v = report["filename_metadata_validation"]
    print(f"Arabic:  expected ~{v['expected_ar_pct']}%, actual {v['actual_ar_user_pct']}%")
    print(f"MCQ:     expected ~{v['expected_mcq_pct']}%, actual {v['actual_mcq_pct']}%")
    print(f"CoT:     expected ~{v['expected_cot_pct']}%, actual {v['actual_think_pct']}%")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming profiler for Arabic fine-tuning JSONL corpus")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to input JSONL file (default: %(default)s)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Directory for output reports (default: %(default)s)")
    args = parser.parse_args()
    profile(input_file=args.input, report_dir=args.report_dir)
