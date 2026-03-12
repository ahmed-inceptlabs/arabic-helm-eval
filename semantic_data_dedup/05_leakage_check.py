#!/usr/bin/env python3
"""Benchmark leakage / contamination check.

Streams the deduped JSONL and flags rows that look like they may have been
derived from one of the six HELM Arabic benchmarks evaluated by this project:

  aratrust, arabic_mmlu, alghafa, arabic_exams, arabic_mmmlu, alrage

Detection strategy (all heuristic / pattern-based):
  1. Structural fingerprints: option formatting, system prompts, scoring rubrics
  2. Keyword anchors: benchmark-specific terms appearing in user or assistant text
  3. Direct benchmark mentions: rows that name the benchmark by name
  4. Answer-pattern leakage: assistant text that reproduces scoring/rubric language

Outputs:
  reports/05_leakage_report.json  — per-benchmark flagged counts + sample rows
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "03_deduped.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Benchmark fingerprints
# ---------------------------------------------------------------------------

BENCHMARK_PATTERNS = {
    "aratrust": {
        "description": "AraTrust safety/ethics benchmark",
        "user_patterns": [
            re.compile(r"(هل يمكنك|اكتب لي).*(إساءة|عنصري|تحرش|خطاب كراهية|عنف|تمييز)", re.DOTALL),
            re.compile(r"(toxicity|bias|offensive|harmful|hate speech)", re.IGNORECASE),
            re.compile(r"(خصوصية|بيانات شخصية|privacy|personal data)", re.IGNORECASE),
            re.compile(r"(أخلاقيات|ethical|morality).*(ذكاء اصطناعي|AI|artificial)", re.IGNORECASE | re.DOTALL),
        ],
        "asst_patterns": [
            re.compile(r"(لا أستطيع|لا يمكنني|أرفض).*(مساعدتك في|تقديم).*(ضار|عنصري|مسيء)", re.DOTALL),
        ],
    },
    "arabic_mmlu": {
        "description": "Arabic MMLU (translated MMLU subjects)",
        "user_patterns": [
            re.compile(
                r"(anatomy|biology|chemistry|physics|mathematics|algebra|history|psychology|"
                r"law|medicine|economics|sociology|philosophy|computer science|astronomy|"
                r"التشريح|الأحياء|الكيمياء|الفيزياء|الرياضيات|التاريخ|علم النفس|"
                r"القانون|الطب|الاقتصاد|علم الاجتماع|الفلسفة|علوم الحاسب|الفلك)",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "alghafa": {
        "description": "AlGhafa NLU benchmark (sentiment, facts, dialects)",
        "user_patterns": [
            re.compile(r"رأي (سلبي|إيجابي|محايد)", re.IGNORECASE),
            re.compile(r"(sentiment|إيجابي جدًا|سلبي جدًا)", re.IGNORECASE),
            re.compile(r"(صحيح|خطأ)\s*$", re.MULTILINE),
            re.compile(r"اختر رقمًا واحدًا فقط من الخيارات أعلاه", re.IGNORECASE),
        ],
        "asst_patterns": [],
    },
    "arabic_exams": {
        "description": "Arabic school exam questions",
        "user_patterns": [
            re.compile(r"(امتحان|اختبار|سؤال.*اختيار|مقرر|منهج|curriculum)", re.IGNORECASE),
            re.compile(r"(exam|test question)", re.IGNORECASE),
            re.compile(
                r"السؤال التالي هو سؤال متعدد الإختيارات",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "arabic_mmmlu": {
        "description": "Arabic Massive MMLU (57 subjects)",
        "user_patterns": [
            re.compile(
                r"(abstract_algebra|clinical_knowledge|college_biology|"
                r"econometrics|electrical_engineering|formal_logic|"
                r"jurisprudence|machine_learning|virology|world_religions)",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "alrage": {
        "description": "AlRAGE generation benchmark",
        "user_patterns": [
            re.compile(r"(اشرح بالتفصيل|وضح بالتفصيل|اكتب مقالا عن|explain in detail)", re.IGNORECASE),
            re.compile(r"(لخص النص التالي|summarize the following)", re.IGNORECASE),
        ],
        "asst_patterns": [],
    },
}

DIRECT_MENTION_PATTERN = re.compile(
    r"\b(aratrust|ara.?trust|arabic.?mmlu|alghafa|al.?ghafa|arabic.?exams|"
    r"arabic.?mmmlu|alrage|al.?rage|HELM|benchmark|leaderboard)\b",
    re.IGNORECASE,
)

RUBRIC_LEAK_PATTERNS = [
    re.compile(r"(الدرجة|score|الإجابة النموذجية|model answer|rubric)", re.IGNORECASE),
    re.compile(r"(الإجابة الصحيحة هي|الجواب الصحيح هو|The correct answer is)", re.IGNORECASE),
]

MCQ_OPTION_PATTERN = re.compile(r"\n\s*[أبجدهـA-E][\)\.]\s")


def extract_texts(row: dict) -> tuple:
    msgs = row.get("messages", [])
    user_text = ""
    asst_text = ""
    sys_text = ""
    for m in msgs:
        role = m.get("role", "")
        if role == "user":
            user_text = m.get("content", "")
        elif role == "assistant":
            asst_text = m.get("content", "")
        elif role == "system":
            sys_text = m.get("content", "")
    asst_clean = THINK_PATTERN.sub("", asst_text)
    return sys_text, user_text, asst_text, asst_clean


def check_leakage(input_file=None, report_dir=None):
    INPUT_FILE = Path(input_file) if input_file else DEFAULT_INPUT
    REPORT_DIR = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_flags = {name: [] for name in BENCHMARK_PATTERNS}
    direct_mention_rows = []
    rubric_leak_rows = []
    mcq_option_count = 0
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            sys_text, user_text, asst_text, asst_clean = extract_texts(row)
            combined = user_text + " " + asst_clean

            if MCQ_OPTION_PATTERN.search(user_text):
                mcq_option_count += 1

            for bname, bconf in BENCHMARK_PATTERNS.items():
                matched = False
                for pat in bconf["user_patterns"]:
                    if pat.search(user_text):
                        matched = True
                        break
                if not matched:
                    for pat in bconf.get("asst_patterns", []):
                        if pat.search(asst_clean):
                            matched = True
                            break
                if matched:
                    benchmark_flags[bname].append(idx)

            if DIRECT_MENTION_PATTERN.search(combined):
                direct_mention_rows.append(idx)

            for pat in RUBRIC_LEAK_PATTERNS:
                if pat.search(asst_clean):
                    rubric_leak_rows.append(idx)
                    break

            if total % 50000 == 0:
                print(f"  ... checked {total:,} rows", file=sys.stderr)

    report = {
        "total_rows": total,
        "mcq_option_rows": mcq_option_count,
        "direct_benchmark_mentions": {
            "count": len(direct_mention_rows),
            "sample_indices": direct_mention_rows[:100],
        },
        "rubric_leak_rows": {
            "count": len(rubric_leak_rows),
            "sample_indices": rubric_leak_rows[:100],
        },
        "benchmark_flags": {},
    }

    all_flagged = set()
    for bname in BENCHMARK_PATTERNS:
        flags = benchmark_flags[bname]
        all_flagged.update(flags)
        report["benchmark_flags"][bname] = {
            "description": BENCHMARK_PATTERNS[bname]["description"],
            "flagged_count": len(flags),
            "flagged_fraction": round(len(flags) / max(total, 1), 4),
            "sample_indices": flags[:50],
        }

    report["summary"] = {
        "total_flagged_unique": len(all_flagged),
        "total_flagged_fraction": round(len(all_flagged) / max(total, 1), 4),
        "note": (
            "These are heuristic pattern matches, not confirmed contamination. "
            "Many flagged rows are legitimately about these topics. "
            "Direct benchmark mentions and rubric leaks are higher-confidence signals."
        ),
    }

    # Collect a few sample rows for the most-flagged benchmarks
    sample_rows = {}
    top_benchmarks = sorted(
        report["benchmark_flags"].items(),
        key=lambda x: x[1]["flagged_count"],
        reverse=True,
    )[:3]

    needed = set()
    for bname, info in top_benchmarks:
        needed.update(info["sample_indices"][:10])

    if needed:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in needed:
                    try:
                        row = json.loads(line.strip())
                        _, user_text, _, _ = extract_texts(row)
                        sample_rows[idx] = user_text[:500]
                    except json.JSONDecodeError:
                        pass
                if len(sample_rows) == len(needed):
                    break

    for bname, info in top_benchmarks:
        previews = []
        for i in info["sample_indices"][:10]:
            if i in sample_rows:
                previews.append({"row_index": i, "user_preview": sample_rows[i]})
        report["benchmark_flags"][bname]["sample_previews"] = previews

    report_path = REPORT_DIR / "05_leakage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK SUMMARY")
    print("=" * 60)
    print(f"Total rows:           {total:,}")
    print(f"MCQ-formatted rows:   {mcq_option_count:,}")
    print(f"Direct mentions:      {len(direct_mention_rows):,}")
    print(f"Rubric leak rows:     {len(rubric_leak_rows):,}")
    print(f"\nPer-benchmark flags:")
    for bname, info in report["benchmark_flags"].items():
        count = info["flagged_count"]
        pct = info["flagged_fraction"] * 100
        print(f"  {bname:20s}: {count:>7,} ({pct:.1f}%)")
    print(f"\nTotal unique flagged: {len(all_flagged):,} "
          f"({100*len(all_flagged)/max(total,1):.1f}%)")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark leakage / contamination check")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to deduped JSONL file (default: %(default)s)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Directory for reports (default: %(default)s)")
    args = parser.parse_args()
    check_leakage(input_file=args.input, report_dir=args.report_dir)
