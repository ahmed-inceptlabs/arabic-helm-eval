#!/usr/bin/env python3
"""Build the final clean dataset by combining all dedup and leakage outputs.

Reads:
  - Source JSONL (original 211k rows)
  - 03_exact_dedup report (exact duplicate indices)
  - 04_semantic_dedup output (semantic cluster removal)
  - 05_leakage report (flagged row indices)

Writes:
  - data/06_clean.jsonl             — final deduplicated dataset
  - data/06_removal_log.jsonl       — every removed row with reason
  - reports/06_final_report.md      — human-readable summary
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SOURCE = SCRIPT_DIR / "ar93_en7_mcq85_open15_cot96_211k.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"


def load_json_safe(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_exact_dedup_removals(data_dir):
    """Get row indices removed by exact dedup (L3 level)."""
    ids_path = data_dir / "03_exact_dedup_ids.json"
    data = load_json_safe(ids_path)
    if not data:
        return set()

    removed = set()
    for level_name in ["L3_think_stripped_pair"]:
        for cluster in data.get(level_name, []):
            indices = cluster.get("row_indices", [])
            if len(indices) > 1:
                for idx in indices[1:]:
                    removed.add(idx)
    return removed


def load_semantic_dedup_removals(data_dir):
    """Get row indices removed by semantic dedup."""
    cluster_path = data_dir / "04_semantic_clusters.json"
    data = load_json_safe(cluster_path)
    if not data:
        return set(), {}

    removed = set()
    stats = {"clusters": len(data), "total_in_clusters": 0, "removed": 0}
    for cluster in data:
        members = [cluster["representative"]] + cluster.get("removed", [])
        stats["total_in_clusters"] += len(members)
        for idx in cluster.get("removed", []):
            removed.add(idx)
    stats["removed"] = len(removed)
    return removed, stats


def load_leakage_flags(report_dir):
    """Get row indices flagged for potential leakage."""
    report_path = report_dir / "05_leakage_report.json"
    data = load_json_safe(report_path)
    if not data:
        return set(), set(), {}

    all_flagged = set()
    high_confidence = set()

    for bname, info in data.get("benchmark_flags", {}).items():
        for idx in info.get("sample_indices", []):
            all_flagged.add(idx)

    for idx in data.get("direct_benchmark_mentions", {}).get("sample_indices", []):
        high_confidence.add(idx)
    for idx in data.get("rubric_leak_rows", {}).get("sample_indices", []):
        high_confidence.add(idx)

    stats = {
        "total_flagged": data.get("summary", {}).get("total_flagged_unique", 0),
        "direct_mentions": data.get("direct_benchmark_mentions", {}).get("count", 0),
        "rubric_leaks": data.get("rubric_leak_rows", {}).get("count", 0),
    }
    return all_flagged, high_confidence, stats


def build_clean(source_file=None, report_dir=None, data_dir=None):
    SOURCE_FILE = Path(source_file) if source_file else DEFAULT_SOURCE
    REPORT_DIR = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    DATA_DIR = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading removal sets...", file=sys.stderr)
    exact_removed = load_exact_dedup_removals(DATA_DIR)
    semantic_removed, semantic_stats = load_semantic_dedup_removals(DATA_DIR)
    leakage_all, leakage_high, leakage_stats = load_leakage_flags(REPORT_DIR)

    print(f"  Exact dedup removals:    {len(exact_removed):,}", file=sys.stderr)
    print(f"  Semantic dedup removals: {len(semantic_removed):,}", file=sys.stderr)
    print(f"  Leakage high-confidence: {len(leakage_high):,}", file=sys.stderr)

    clean_path = DATA_DIR / "06_clean.jsonl"
    log_path = DATA_DIR / "06_removal_log.jsonl"

    total = 0
    kept = 0
    reason_counts = Counter()

    with open(SOURCE_FILE, "r", encoding="utf-8") as fin, \
         open(clean_path, "w", encoding="utf-8") as fout, \
         open(log_path, "w", encoding="utf-8") as flog:

        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            total += 1

            reasons = []
            if idx in exact_removed:
                reasons.append("exact_duplicate")
            if idx in semantic_removed:
                reasons.append("semantic_duplicate")
            if idx in leakage_high:
                reasons.append("leakage_high_confidence")

            if reasons:
                for r in reasons:
                    reason_counts[r] += 1
                flog.write(json.dumps({
                    "row_index": idx,
                    "reasons": reasons,
                }, ensure_ascii=False) + "\n")
            else:
                fout.write(line + "\n")
                kept += 1

            if total % 50000 == 0:
                print(f"  ... processed {total:,}, kept {kept:,}", file=sys.stderr)

    removed = total - kept

    # Load profile and dedup reports for the summary
    profile = load_json_safe(REPORT_DIR / "01_profile_report.json") or {}
    exact_report = load_json_safe(REPORT_DIR / "03_exact_dedup_report.json") or {}
    sim_hist = load_json_safe(REPORT_DIR / "04_similarity_histogram.json") or {}
    leakage_report = load_json_safe(REPORT_DIR / "05_leakage_report.json") or {}

    # Build markdown report
    md = []
    md.append("# Dataset Quality & Deduplication Report\n")
    md.append(f"**Source:** `{SOURCE_FILE.name}`\n")
    md.append(f"**Total rows:** {total:,}\n")
    md.append(f"**Final clean rows:** {kept:,}\n")
    md.append(f"**Total removed:** {removed:,} ({100*removed/max(total,1):.2f}%)\n")

    md.append("\n## 1. Profile Summary\n")
    if profile:
        v = profile.get("filename_metadata_validation", {})
        md.append(f"- Arabic user prompts: {v.get('actual_ar_user_pct', '?')}% (expected ~{v.get('expected_ar_pct', '?')}%)\n")
        md.append(f"- MCQ fraction: {v.get('actual_mcq_pct', '?')}% (expected ~{v.get('expected_mcq_pct', '?')}%)\n")
        md.append(f"- CoT (<think>) fraction: {v.get('actual_think_pct', '?')}% (expected ~{v.get('expected_cot_pct', '?')}%)\n")

        ul = profile.get("user_length", {})
        al = profile.get("assistant_length", {})
        md.append(f"- User prompt length: mean={ul.get('mean', '?')}, min={ul.get('min', '?')}, max={ul.get('max', '?')}\n")
        md.append(f"- Assistant length: mean={al.get('mean', '?')}, min={al.get('min', '?')}, max={al.get('max', '?')}\n")

    md.append("\n## 2. Exact Dedup (Phase 3)\n")
    if exact_report:
        for lname, info in exact_report.get("levels", {}).items():
            md.append(f"- **{lname}**: {info.get('unique_hashes', '?'):,} unique, "
                      f"{info.get('duplicate_rows_removable', '?'):,} removable "
                      f"({info.get('duplicate_fraction', 0)*100:.2f}%)\n")
        dedup_out = exact_report.get("dedup_output", {})
        md.append(f"- Output (L3): kept {dedup_out.get('kept', '?'):,}, removed {dedup_out.get('removed', '?'):,}\n")

    md.append("\n## 3. Semantic Dedup (Phase 4)\n")
    if semantic_stats:
        md.append(f"- Clusters: {semantic_stats.get('clusters', '?'):,}\n")
        md.append(f"- Rows in clusters: {semantic_stats.get('total_in_clusters', '?'):,}\n")
        md.append(f"- Removable: {semantic_stats.get('removed', '?'):,}\n")
    if sim_hist:
        stats = sim_hist.get("stats", {})
        md.append(f"- Similarity stats: mean={stats.get('mean', '?')}, "
                  f"median={stats.get('median', '?')}, "
                  f"p95={stats.get('p95', '?')}, "
                  f"p99={stats.get('p99', '?')}\n")
        md.append("\n| Band | Count |\n|------|-------|\n")
        for band, count in sim_hist.get("histogram", {}).items():
            md.append(f"| {band} | {count:,} |\n")

    md.append("\n## 4. Leakage Check (Phase 5)\n")
    if leakage_report:
        md.append(f"- Direct benchmark mentions: {leakage_stats.get('direct_mentions', '?'):,}\n")
        md.append(f"- Rubric leak rows: {leakage_stats.get('rubric_leaks', '?'):,}\n")
        md.append(f"- Total flagged (heuristic): {leakage_stats.get('total_flagged', '?'):,}\n")
        md.append("\n| Benchmark | Flagged | % |\n|-----------|---------|---|\n")
        for bname, info in leakage_report.get("benchmark_flags", {}).items():
            md.append(f"| {bname} | {info.get('flagged_count', 0):,} | "
                      f"{info.get('flagged_fraction', 0)*100:.1f}% |\n")

    md.append("\n## 5. Final Removal Breakdown\n")
    md.append("\n| Reason | Rows Removed |\n|--------|--------------|\n")
    for reason, count in reason_counts.most_common():
        md.append(f"| {reason} | {count:,} |\n")
    md.append(f"| **Total unique removed** | **{removed:,}** |\n")

    md.append(f"\n## 6. Output Files\n")
    md.append(f"- `data/06_clean.jsonl` — {kept:,} rows\n")
    md.append(f"- `data/06_removal_log.jsonl` — {removed:,} removal entries\n")

    report_md = "".join(md)
    report_path = REPORT_DIR / "06_final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 60)
    print("FINAL BUILD SUMMARY")
    print("=" * 60)
    print(f"Source rows:   {total:,}")
    print(f"Kept:          {kept:,}")
    print(f"Removed:       {removed:,} ({100*removed/max(total,1):.2f}%)")
    print(f"\nRemoval reasons:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:30s}: {count:,}")
    print(f"\nClean dataset: {clean_path}")
    print(f"Removal log:   {log_path}")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build final clean dataset from all dedup and leakage outputs")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Path to original source JSONL (default: %(default)s)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Directory containing reports from previous steps (default: %(default)s)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory containing data from previous steps (default: %(default)s)")
    args = parser.parse_args()
    build_clean(source_file=args.source, report_dir=args.report_dir, data_dir=args.data_dir)
