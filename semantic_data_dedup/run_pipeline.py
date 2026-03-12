#!/usr/bin/env python3
"""Run the full dedup pipeline end-to-end.

Usage:
    # Defaults (OpenAI backend, threshold 0.93)
    python run_pipeline.py

    # Custom input and threshold
    python run_pipeline.py --input my_data.jsonl --threshold 0.96

    # Skip to a specific step (e.g., re-run from semantic dedup)
    python run_pipeline.py --start-from 4

    # Local embeddings, no API key needed
    python run_pipeline.py --backend local

    # Only run steps 1-3 (no embeddings)
    python run_pipeline.py --stop-after 3
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "ar93_en7_mcq85_open15_cot96_211k.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "reports" / "02_samples"


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def run_step(step_num, name, fn, args_dict):
    print(f"\n{'='*60}", flush=True)
    print(f"  STEP {step_num}: {name}", flush=True)
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    fn(**args_dict)
    elapsed = time.time() - t0
    print(f"\n  [Step {step_num} completed in {fmt_time(elapsed)}]", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full semantic dedup pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Profile         Streaming stats, language mix, MCQ detection
  2  Sample          Stratified reservoir sampling for manual audit
  3  Exact Dedup     Hash-based dedup at 5 normalization levels (L1-L5)
  4  Semantic Dedup  OpenAI/local embeddings + FAISS cosine similarity
  5  Leakage Check   Benchmark contamination pattern matching
  6  Build Clean     Combine all removals into final clean dataset

Examples:
  python run_pipeline.py
  python run_pipeline.py --input data.jsonl --threshold 0.96
  python run_pipeline.py --start-from 4 --skip-embed
  python run_pipeline.py --stop-after 3
  python run_pipeline.py --backend local
""",
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--input", default=str(DEFAULT_INPUT),
                          help="Source JSONL file (default: %(default)s)")
    io_group.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                          help="Reports output directory (default: %(default)s)")
    io_group.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                          help="Data output directory (default: %(default)s)")

    sample_group = parser.add_argument_group("Step 2: Sampling")
    sample_group.add_argument("--sample-size", type=int, default=50,
                              help="Rows per sample slice (default: %(default)s)")
    sample_group.add_argument("--seed", type=int, default=42,
                              help="Random seed for sampling (default: %(default)s)")

    sem_group = parser.add_argument_group("Step 4: Semantic Dedup")
    sem_group.add_argument("--backend", choices=["openai", "local"], default="openai",
                           help="Embedding backend (default: %(default)s)")
    sem_group.add_argument("--threshold", type=float, default=0.93,
                           help="Cosine similarity threshold for clustering (default: %(default)s)")
    sem_group.add_argument("--top-k", type=int, default=10,
                           help="Nearest neighbors to retrieve per row (default: %(default)s)")
    sem_group.add_argument("--skip-embed", action="store_true",
                           help="Reuse cached embeddings (skip embedding + FAISS search)")

    flow_group = parser.add_argument_group("Pipeline Control")
    flow_group.add_argument("--start-from", type=int, default=1, choices=range(1, 7),
                            metavar="{1-6}",
                            help="Start from this step number (default: 1)")
    flow_group.add_argument("--stop-after", type=int, default=6, choices=range(1, 7),
                            metavar="{1-6}",
                            help="Stop after this step number (default: 6)")

    args = parser.parse_args()

    if args.start_from > args.stop_after:
        print(f"Error: --start-from ({args.start_from}) > --stop-after ({args.stop_after})")
        sys.exit(1)

    input_path = str(args.input)
    report_dir = str(args.report_dir)
    data_dir = str(args.data_dir)
    sample_dir = str(Path(report_dir) / "02_samples")
    deduped_file = str(Path(data_dir) / "03_deduped.jsonl")

    print(f"Semantic Dedup Pipeline", flush=True)
    print(f"  Input:      {input_path}", flush=True)
    print(f"  Reports:    {report_dir}", flush=True)
    print(f"  Data:       {data_dir}", flush=True)
    print(f"  Steps:      {args.start_from} → {args.stop_after}", flush=True)
    if args.start_from <= 4 <= args.stop_after:
        print(f"  Backend:    {args.backend}", flush=True)
        print(f"  Threshold:  {args.threshold}", flush=True)
        print(f"  Top-k:      {args.top_k}", flush=True)
        print(f"  Skip embed: {args.skip_embed}", flush=True)

    total_t0 = time.time()
    timings = {}

    # --- Step 1: Profile ---
    if args.start_from <= 1 <= args.stop_after:
        from importlib import import_module
        mod = __import__("01_profile")
        timings[1] = run_step(1, "Profile", mod.profile, {
            "input_file": input_path,
            "report_dir": report_dir,
        })

    # --- Step 2: Sample ---
    if args.start_from <= 2 <= args.stop_after:
        mod = __import__("02_sample")
        timings[2] = run_step(2, "Stratified Sampling", mod.sample, {
            "input_file": input_path,
            "sample_dir": sample_dir,
            "sample_size": args.sample_size,
            "seed": args.seed,
        })

    # --- Step 3: Exact Dedup ---
    if args.start_from <= 3 <= args.stop_after:
        mod = __import__("03_exact_dedup")
        timings[3] = run_step(3, "Exact Dedup (L1-L5)", mod.dedup, {
            "input_file": input_path,
            "report_dir": report_dir,
            "data_dir": data_dir,
        })

    # --- Step 4: Semantic Dedup ---
    if args.start_from <= 4 <= args.stop_after:
        mod = __import__("04_semantic_dedup")
        sys.argv = [
            "04_semantic_dedup.py",
            "--backend", args.backend,
            "--threshold", str(args.threshold),
            "--top-k", str(args.top_k),
        ]
        if args.skip_embed:
            sys.argv.append("--skip-embed")
        timings[4] = run_step(4, "Semantic Dedup", mod.main, {})

    # --- Step 5: Leakage Check ---
    if args.start_from <= 5 <= args.stop_after:
        mod = __import__("05_leakage_check")
        timings[5] = run_step(5, "Leakage Check", mod.check_leakage, {
            "input_file": deduped_file,
            "report_dir": report_dir,
        })

    # --- Step 6: Build Clean ---
    if args.start_from <= 6 <= args.stop_after:
        mod = __import__("06_build_clean")
        timings[6] = run_step(6, "Build Clean Dataset", mod.build_clean, {
            "source_file": input_path,
            "report_dir": report_dir,
            "data_dir": data_dir,
        })

    total_elapsed = time.time() - total_t0

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nStep timings:")
    step_names = {
        1: "Profile", 2: "Sampling", 3: "Exact Dedup",
        4: "Semantic Dedup", 5: "Leakage Check", 6: "Build Clean",
    }
    for step, elapsed in sorted(timings.items()):
        print(f"  {step}. {step_names[step]:20s} {fmt_time(elapsed)}")
    print(f"\n  Total: {fmt_time(total_elapsed)}")


if __name__ == "__main__":
    main()
