#!/usr/bin/env python3
"""Exact and normalized deduplication at five normalization levels.

Streams the JSONL, hashes each row at multiple normalization levels, reports
duplicate counts, and writes a deduplicated JSONL (removing exact duplicates
at the most aggressive normalization level).

Normalization levels:
  1. Raw pair: user + assistant as-is
  2. Whitespace-normalized pair
  3. Think-stripped pair: remove <think>...</think>, then whitespace-normalize
  4. User-only: whitespace-normalized user prompt
  5. User-only with Arabic canonicalization: alef/ya/ta-marbuta normalization,
     strip diacritics (tashkeel), unify punctuation
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "ar93_en7_mcq85_open15_cot96_211k.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TASHKEEL = re.compile(r"[\u064B-\u065F\u0670]")
WHITESPACE = re.compile(r"\s+")

ALEF_MAP = str.maketrans({
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
})

YA_TA_MAP = str.maketrans({
    "\u0649": "\u064A",  # ى -> ي
    "\u0629": "\u0647",  # ة -> ه
})

PUNCT_MAP = str.maketrans({
    "\u060C": ",",   # ،
    "\u061B": ";",   # ؛
    "\u061F": "?",   # ؟
    "\u066B": ".",   # ٫
    "\u066C": ",",   # ٬
    "\u06D4": ".",   # ۔
    "\u200F": "",    # RTL mark
    "\u200E": "",    # LTR mark
    "\u200B": "",    # zero-width space
    "\u00A0": " ",   # non-breaking space
    "\uFEFF": "",    # BOM
})


def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def strip_think(text: str) -> str:
    return THINK_PATTERN.sub("", text)


def arabic_canonicalize(text: str) -> str:
    text = text.translate(ALEF_MAP)
    text = text.translate(YA_TA_MAP)
    text = text.translate(PUNCT_MAP)
    text = TASHKEEL.sub("", text)
    text = normalize_whitespace(text)
    return text.lower()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_texts(row: dict) -> tuple:
    msgs = row.get("messages", [])
    user_text = ""
    asst_text = ""
    for m in msgs:
        if m.get("role") == "user":
            user_text = m.get("content", "")
        elif m.get("role") == "assistant":
            asst_text = m.get("content", "")
    return user_text, asst_text


def dedup(input_file=None, report_dir=None, data_dir=None):
    INPUT_FILE = Path(input_file) if input_file else DEFAULT_INPUT
    REPORT_DIR = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    DATA_DIR = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    levels = {
        "L1_raw_pair": {},
        "L2_ws_pair": {},
        "L3_think_stripped_pair": {},
        "L4_user_only": {},
        "L5_user_arabic_canon": {},
    }

    hash_to_first_idx = {name: {} for name in levels}
    cluster_map = {name: defaultdict(list) for name in levels}

    total = 0
    print("Pass 1: Computing hashes...", file=sys.stderr)

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
            user_text, asst_text = extract_texts(row)

            l1 = sha256(user_text + "\x00" + asst_text)
            ws_user = normalize_whitespace(user_text)
            ws_asst = normalize_whitespace(asst_text)
            l2 = sha256(ws_user + "\x00" + ws_asst)
            ts_asst = normalize_whitespace(strip_think(asst_text))
            l3 = sha256(ws_user + "\x00" + ts_asst)
            l4 = sha256(ws_user)
            l5 = sha256(arabic_canonicalize(user_text))

            hashes = {
                "L1_raw_pair": l1,
                "L2_ws_pair": l2,
                "L3_think_stripped_pair": l3,
                "L4_user_only": l4,
                "L5_user_arabic_canon": l5,
            }

            for name, h in hashes.items():
                if h not in hash_to_first_idx[name]:
                    hash_to_first_idx[name][h] = idx
                cluster_map[name][h].append(idx)

            if total % 50000 == 0:
                print(f"  ... hashed {total:,} rows", file=sys.stderr)

    print(f"Total rows: {total:,}", file=sys.stderr)

    report = {"total_rows": total, "levels": {}}
    for name in levels:
        clusters = cluster_map[name]
        unique = sum(1 for v in clusters.values() if len(v) == 1)
        dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        dup_rows = sum(len(v) - 1 for v in dup_clusters.values())
        total_in_dup_clusters = sum(len(v) for v in dup_clusters.values())

        top_repeated = sorted(dup_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:20]

        report["levels"][name] = {
            "unique_hashes": len(clusters),
            "unique_singletons": unique,
            "duplicate_clusters": len(dup_clusters),
            "duplicate_rows_removable": dup_rows,
            "duplicate_fraction": round(dup_rows / max(total, 1), 4),
            "top_repeated_sizes": [len(v) for _, v in top_repeated],
        }

        print(f"\n{name}:", file=sys.stderr)
        print(f"  Unique hashes: {len(clusters):,}", file=sys.stderr)
        print(f"  Duplicate clusters: {len(dup_clusters):,}", file=sys.stderr)
        print(f"  Removable duplicates: {dup_rows:,} ({100*dup_rows/total:.1f}%)", file=sys.stderr)

    # Collect top-20 most-repeated user prompts for the report (L5 level)
    top_templates = []
    l5_clusters = cluster_map["L5_user_arabic_canon"]
    l5_sorted = sorted(l5_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        needed_indices = set()
        for _, indices in l5_sorted:
            needed_indices.add(indices[0])

        idx_to_text = {}
        for idx, line in enumerate(f):
            if idx in needed_indices:
                try:
                    row = json.loads(line.strip())
                    user_text, _ = extract_texts(row)
                    idx_to_text[idx] = user_text[:500]
                except json.JSONDecodeError:
                    pass
            if len(idx_to_text) == len(needed_indices):
                break

    for h, indices in l5_sorted:
        first_idx = indices[0]
        top_templates.append({
            "count": len(indices),
            "first_row_index": first_idx,
            "user_preview": idx_to_text.get(first_idx, ""),
        })

    report["top_repeated_templates_L5"] = top_templates

    # Write deduped file using L3 (think-stripped pair) as the primary dedup key
    # This removes rows that are identical after stripping <think> blocks
    print("\nPass 2: Writing deduplicated file (L3 level)...", file=sys.stderr)
    l3_seen = set()
    kept = 0
    removed = 0

    deduped_path = DATA_DIR / "03_deduped.jsonl"
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(deduped_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_text, asst_text = extract_texts(row)
            ws_user = normalize_whitespace(user_text)
            ts_asst = normalize_whitespace(strip_think(asst_text))
            h = sha256(ws_user + "\x00" + ts_asst)

            if h not in l3_seen:
                l3_seen.add(h)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1

            if (idx + 1) % 50000 == 0:
                print(f"  ... processed {idx+1:,}, kept {kept:,}, removed {removed:,}", file=sys.stderr)

    report["dedup_output"] = {
        "level_used": "L3_think_stripped_pair",
        "kept": kept,
        "removed": removed,
        "output_file": str(deduped_path.relative_to(SCRIPT_DIR)),
    }

    # Save cluster IDs for all levels
    cluster_ids = {}
    for name in levels:
        dup_clusters = {k: v for k, v in cluster_map[name].items() if len(v) > 1}
        cluster_ids[name] = [
            {"hash": k, "row_indices": v}
            for k, v in sorted(dup_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:200]
        ]

    ids_path = DATA_DIR / "03_exact_dedup_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(cluster_ids, f, ensure_ascii=False, indent=2)

    report_path = REPORT_DIR / "03_exact_dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nDeduped file: {deduped_path} ({kept:,} rows)", file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)
    print(f"Cluster IDs: {ids_path}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("EXACT DEDUP SUMMARY")
    print("=" * 60)
    print(f"Total rows:  {total:,}")
    for name, info in report["levels"].items():
        print(f"\n{name}:")
        print(f"  Unique:     {info['unique_hashes']:,}")
        print(f"  Removable:  {info['duplicate_rows_removable']:,} "
              f"({info['duplicate_fraction']*100:.1f}%)")
    print(f"\nOutput (L3 dedup): {kept:,} kept, {removed:,} removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact and normalized deduplication at five normalization levels")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to input JSONL file (default: %(default)s)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Directory for reports (default: %(default)s)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory for data outputs (default: %(default)s)")
    args = parser.parse_args()
    dedup(input_file=args.input, report_dir=args.report_dir, data_dir=args.data_dir)
