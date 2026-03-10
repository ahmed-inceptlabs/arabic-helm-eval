#!/usr/bin/env python3
"""Store HELM evaluation results into the edubench database incrementally (batches of 10)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import ijson
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

BATCH_SIZE = 10


def get_git_info():
    """Get current git commit SHA and branch name."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
        return commit, branch
    except subprocess.CalledProcessError:
        return None, None


def get_db_connection():
    """Connect to the edubench database using .env credentials."""
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432),
    )


def load_json(path):
    """Load a JSON file, decoding unicode escapes to real characters."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_instance_stats_lookup(per_instance_stats):
    """Build a dict: instance_id -> {exact_match, num_completion_tokens, ...}."""
    lookup = {}
    for entry in per_instance_stats:
        iid = entry["instance_id"]
        if iid not in lookup:
            lookup[iid] = {}
        for stat in entry.get("stats", []):
            name = stat["name"]["name"]
            if name in ("exact_match", "num_completion_tokens", "inference_runtime", "alrage_score"):
                lookup[iid][name] = stat.get("mean", stat.get("sum", 0))
    return lookup


def extract_correct_reference(references):
    """Find the reference tagged 'correct' and return its text."""
    for ref in references:
        if "correct" in ref.get("tags", []):
            return ref["output"]["text"]
    return None


def detect_format(request_state):
    """Detect format: 'generation' for open-ended QA, 'MCQ_N' for multiple choice."""
    output_mapping = request_state.get("output_mapping", {})
    if not output_mapping:
        return "generation"
    references = request_state.get("instance", {}).get("references", [])
    return f"MCQ_{len(references)}"


def insert_run(cur, run_spec, stats, suite, git_commit, git_branch):
    """Insert an eval_runs row and return the new run_id."""
    adapter = run_spec.get("adapter_spec", {})
    model_name = adapter.get("model", "unknown")

    # Extract task name from scenario_spec
    scenario_class = run_spec.get("scenario_spec", {}).get("class_name", "")
    task_name = scenario_class.split(".")[-1].replace("Scenario", "").lower()
    if not task_name:
        task_name = run_spec.get("name", "unknown").split(":")[0]

    # Build aggregated metrics from stats.json
    metrics = {}
    for stat in stats:
        name = stat["name"]["name"]
        if name in ("exact_match", "quasi_exact_match", "prefix_exact_match", "alrage_score"):
            metrics[name] = stat.get("mean", stat.get("sum"))

    # Build config with git info and suite
    config = {
        "suite": suite,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "adapter_spec": adapter,
        "scenario_spec": run_spec.get("scenario_spec", {}),
        "helm_run_name": run_spec.get("name", ""),
    }

    # Determine if thinking model
    is_thinking = "thinking" in model_name.lower() or adapter.get(
        "reasoning_effort", ""
    ) not in ("", "off", None)

    # Determine judge model
    if "alrage" in scenario_class.lower():
        judge_model = "openai/gpt-4o-2024-11-20"
    else:
        judge_model = "helm_exact_match"

    cur.execute(
        """
        INSERT INTO eval_runs (model_name, task_name, judge_model, status, started_at, metrics, config, is_thinking_model)
        VALUES (%s, %s, %s, 'completed', NOW(), %s, %s, %s)
        RETURNING id
        """,
        (
            model_name,
            task_name,
            judge_model,
            json.dumps(metrics, ensure_ascii=False),
            json.dumps(config, ensure_ascii=False),
            is_thinking,
        ),
    )
    return cur.fetchone()[0]


def build_sample_row(run_id, idx, request_state, instance_stats):
    """Build a tuple for inserting into eval_samples."""
    instance = request_state["instance"]
    references = instance.get("references", [])
    instance_id = instance.get("id", f"id{idx}")

    # Question text (real Arabic from json.load)
    instruction = instance.get("input", {}).get("text", "")

    # Correct answer
    reference = extract_correct_reference(references)

    # Model prediction
    result = request_state.get("result", {})
    completions = result.get("completions", [])
    raw_response = completions[0]["text"] if completions else ""

    # Mapped output (e.g. letter -> full answer text)
    output_mapping = request_state.get("output_mapping", {})
    prediction = output_mapping.get(raw_response, raw_response)

    # Prompt
    raw_prompt = request_state.get("request", {}).get("prompt", "")

    # Format
    fmt = detect_format(request_state)

    # Stats from per_instance_stats
    stats = instance_stats.get(instance_id, {})
    judge_score = stats.get("exact_match") if "exact_match" in stats else stats.get("alrage_score")
    answer_tokens = stats.get("num_completion_tokens")
    latency = result.get("request_time")
    latency_ms = round(latency * 1000, 2) if latency else None

    # Meta
    meta = {
        "output_mapping": output_mapping,
        "all_references": [
            {"text": r["output"]["text"], "correct": "correct" in r.get("tags", [])}
            for r in references
        ],
        "finish_reason": completions[0].get("finish_reason", {}).get("reason")
        if completions
        else None,
        "instance_id": instance_id,
    }

    # Include annotation data if present (e.g., ALRAGE GPT-4o judge results)
    annotations = request_state.get("annotations", {})
    if annotations:
        meta["annotations"] = annotations

    # Parse numeric index from instance_id like "id42" -> 42
    try:
        sample_index = int(instance_id.replace("id", ""))
    except ValueError:
        sample_index = idx

    return (
        run_id,
        sample_index,
        fmt,
        instruction,
        reference,
        prediction,
        raw_prompt,
        raw_response,
        judge_score,
        latency_ms,
        answer_tokens,
        json.dumps(meta, ensure_ascii=False),
    )


INSERT_SAMPLES_SQL = """
    INSERT INTO eval_samples
        (run_id, sample_index, format, instruction, reference, prediction,
         raw_prompt, raw_response, judge_score, latency_ms, answer_tokens, meta)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def flush_batch(cur, conn, batch, total_inserted):
    """Insert a batch of samples and commit."""
    cur.executemany(INSERT_SAMPLES_SQL, batch)
    conn.commit()
    total_inserted += len(batch)
    print(f"  Inserted {total_inserted} samples...", end="\r")
    return total_inserted


def stream_and_store_samples(conn, cur, run_id, run_dir, instance_stats):
    """Stream scenario_state.json and insert samples in batches of 10."""
    scenario_path = run_dir / "scenario_state.json"

    batch = []
    total_inserted = 0
    idx = 0

    with open(scenario_path, "rb") as f:
        for request_state in ijson.items(f, "request_states.item"):
            row = build_sample_row(run_id, idx, request_state, instance_stats)
            batch.append(row)
            idx += 1

            if len(batch) >= BATCH_SIZE:
                total_inserted = flush_batch(cur, conn, batch, total_inserted)
                batch = []

    # Flush remaining
    if batch:
        total_inserted = flush_batch(cur, conn, batch, total_inserted)

    print()  # newline after \r progress
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Store HELM results in database")
    parser.add_argument("--run-dir", required=True, help="Path to HELM run output directory")
    parser.add_argument("--suite", default="", help="Suite label for tracking")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    # Check required files
    for fname in ("run_spec.json", "scenario_state.json", "per_instance_stats.json", "stats.json"):
        if not (run_dir / fname).exists():
            print(f"Error: missing {fname} in {run_dir}")
            sys.exit(1)

    print(f"Loading run from: {run_dir}")

    # Step 1: Load small files
    run_spec = load_json(run_dir / "run_spec.json")
    stats = load_json(run_dir / "stats.json")
    per_instance_stats = load_json(run_dir / "per_instance_stats.json")

    # Step 2: Build instance stats lookup
    instance_stats = build_instance_stats_lookup(per_instance_stats)
    print(f"  Loaded stats for {len(instance_stats)} instances")

    # Step 3: Git info
    git_commit, git_branch = get_git_info()
    print(f"  Git: {git_branch}@{git_commit[:8] if git_commit else 'N/A'}")

    # Step 4: Connect to DB
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step 5: Insert eval_runs
        run_id = insert_run(cur, run_spec, stats, args.suite, git_commit, git_branch)
        conn.commit()
        print(f"  Created eval_run id={run_id}")

        # Step 6: Stream and insert samples in batches of 10
        total = stream_and_store_samples(conn, cur, run_id, run_dir, instance_stats)

        # Step 7: Update total_samples on the run
        cur.execute(
            "UPDATE eval_runs SET total_samples = %s, completed_at = NOW() WHERE id = %s",
            (total, run_id),
        )
        conn.commit()

        # Summary
        if any("alrage_score" in s for s in instance_stats.values()):
            scores = [s.get("alrage_score", 0) for s in instance_stats.values()]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"\nDone! run_id={run_id}, samples={total}, avg_score={avg_score:.3f}")
        else:
            correct = sum(1 for s in instance_stats.values() if s.get("exact_match", 0) == 1.0)
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"\nDone! run_id={run_id}, samples={total}, accuracy={accuracy:.1f}%")

    except Exception as e:
        conn.rollback()
        print(f"\nError: {e}")
        # Clean up partial run
        if "run_id" in locals():
            cur.execute("DELETE FROM eval_samples WHERE run_id = %s", (run_id,))
            cur.execute("DELETE FROM eval_runs WHERE id = %s", (run_id,))
            conn.commit()
            print(f"  Cleaned up partial run_id={run_id}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
