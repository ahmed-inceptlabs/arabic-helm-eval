#!/usr/bin/env python3
"""Unified CLI: configure HELM, run evaluation, and store results in DB — all in one command.

Usage:
    # Fireworks model
    python helm_eval.py \\
      --model-name fireworks/kimi-k2p5 \\
      --api-base https://api.fireworks.ai/inference/v1 \\
      --api-model accounts/fireworks/models/kimi-k2p5 \\
      --tokenizer Qwen/Qwen2.5-7B \\
      --benchmark aratrust \\
      --suite my-test \\
      --max-instances 10

    # Local LM Studio model
    python helm_eval.py \\
      --model-name local/my-model \\
      --api-base http://127.0.0.1:1234/v1 \\
      --api-key lm-studio \\
      --tokenizer Qwen/Qwen3.5-9B \\
      --benchmark aratrust \\
      --suite local-test \\
      --max-instances 10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from store_helm_results import (
    build_instance_stats_lookup,
    get_db_connection,
    get_git_info,
    insert_run,
    load_json,
    stream_and_store_samples,
)

PROJECT_DIR = Path(__file__).parent

# Available benchmarks and their default args
BENCHMARKS = {
    "aratrust": "category=all",
    "arabic_mmlu": "subset=all",
    "alghafa": "subset=all",
    "arabic_exams": "subject=all",
    "arabic_mmmlu": "subject=all",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            default_style='"',
            indent=2,
        )


def upsert_list_entry(entries, key_field, key_value, new_entry):
    """Update existing entry by key or append. Returns 'updated' or 'added'."""
    for i, entry in enumerate(entries):
        if entry.get(key_field) == key_value:
            entries[i] = new_entry
            return "updated"
    entries.append(new_entry)
    return "added"


def ensure_model_deployment(args):
    """Add or update model entry in model_deployments.yaml."""
    path = PROJECT_DIR / "model_deployments.yaml"
    data = load_yaml(path)
    deployments = data.setdefault("model_deployments", [])

    if args.api_model:
        client_spec = {
            "class_name": "fireworks_client.FireworksNoThinkingClient",
            "args": {
                "base_url": args.api_base,
                "openai_model_name": args.api_model,
            },
        }
    else:
        client_spec = {
            "class_name": "helm.clients.openai_client.OpenAIClient",
            "args": {
                "api_key": args.api_key or "lm-studio",
                "org_id": "",
                "base_url": args.api_base,
            },
        }

    # Use existing tokenizer_name if model already configured, otherwise derive from model name
    tokenizer_name = args.model_name
    for d in deployments:
        if d.get("name") == args.model_name:
            tokenizer_name = d.get("tokenizer_name", tokenizer_name)
            break

    entry = {
        "name": args.model_name,
        "model_name": args.model_name,
        "tokenizer_name": tokenizer_name,
        "max_sequence_length": args.max_seq_len,
        "client_spec": client_spec,
    }

    action = upsert_list_entry(deployments, "name", args.model_name, entry)
    save_yaml(path, data)
    print(f"  model_deployments.yaml: {action} {args.model_name}")
    return tokenizer_name


def ensure_tokenizer_config(args, tokenizer_name):
    """Add or update tokenizer entry in tokenizer_configs.yaml."""
    path = PROJECT_DIR / "tokenizer_configs.yaml"
    data = load_yaml(path)
    configs = data.setdefault("tokenizer_configs", [])

    entry = {
        "name": tokenizer_name,
        "tokenizer_spec": {
            "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
            "args": {"pretrained_model_name_or_path": args.tokenizer},
        },
        "end_of_text_token": "<|im_end|>",
        "prefix_token": "<|im_start|>",
    }

    action = upsert_list_entry(configs, "name", tokenizer_name, entry)
    save_yaml(path, data)
    print(f"  tokenizer_configs.yaml: {action} {tokenizer_name}")


def ensure_model_metadata(args):
    """Add or update model entry in model_metadata.yaml."""
    path = PROJECT_DIR / "model_metadata.yaml"
    data = load_yaml(path)
    models = data.setdefault("models", [])

    entry = {
        "name": args.model_name,
        "display_name": args.display_name or args.model_name,
        "description": args.display_name or args.model_name,
        "creator_organization_name": args.creator or "",
        "access": "open",
    }

    action = upsert_list_entry(models, "name", args.model_name, entry)
    save_yaml(path, data)
    print(f"  model_metadata.yaml: {action} {args.model_name}")


def generate_run_spec(args):
    """Generate a run_specs conf file for this run."""
    benchmark = args.benchmark
    bench_args = args.benchmark_args or BENCHMARKS.get(benchmark, "")
    description = f"{benchmark}:{bench_args},model={args.model_name}"

    conf_path = PROJECT_DIR / f"run_specs_{args.suite}.conf"
    conf_path.write_text(
        f'entries: [\n  {{description: "{description}", priority: 1}}\n]\n',
        encoding="utf-8",
    )
    print(f"  {conf_path.name}: {description}")
    return conf_path


def update_credentials(args):
    """Update credentials.conf if --api-key is provided."""
    if not args.api_key:
        return

    path = PROJECT_DIR / "credentials.conf"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    key_name = "fireworksApiKey" if args.api_model else "openaiApiKey"
    new_line = f'{key_name}: "{args.api_key}"'

    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key_name}:"):
            lines[i] = new_line
            updated = True
            break
    if not updated:
        lines.append(new_line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  credentials.conf: {'updated' if updated else 'added'} {key_name}")


def run_helm(conf_path, suite, max_instances):
    """Run helm-run as a subprocess."""
    cmd = [
        "helm-run",
        "--conf-paths", str(conf_path),
        "--suite", suite,
        "--local-path", ".",
        "--max-eval-instances", str(max_instances),
    ]
    print(f"\n=== Running HELM ===")
    print(f"  PYTHONPATH=. {' '.join(cmd)}")
    print()

    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f".:{pythonpath}" if pythonpath else "."

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nError: helm-run exited with code {result.returncode}")
        sys.exit(result.returncode)


def store_results(suite):
    """Find run directories and store results in DB."""
    print(f"\n=== Storing Results ===")
    suite_dir = PROJECT_DIR / "benchmark_output" / "runs" / suite

    if not suite_dir.exists():
        print(f"Error: suite directory not found: {suite_dir}")
        sys.exit(1)

    run_dirs = [
        d for d in suite_dir.iterdir()
        if d.is_dir() and (d / "scenario_state.json").exists()
    ]
    if not run_dirs:
        print(f"Error: no completed runs found in {suite_dir}")
        sys.exit(1)

    git_commit, git_branch = get_git_info()

    for run_dir in run_dirs:
        print(f"\nStoring: {run_dir.name}")

        required = ("run_spec.json", "scenario_state.json", "per_instance_stats.json", "stats.json")
        missing = [f for f in required if not (run_dir / f).exists()]
        if missing:
            print(f"  Skipping — missing: {', '.join(missing)}")
            continue

        run_spec = load_json(run_dir / "run_spec.json")
        stats = load_json(run_dir / "stats.json")
        per_instance_stats = load_json(run_dir / "per_instance_stats.json")
        instance_stats = build_instance_stats_lookup(per_instance_stats)
        print(f"  Loaded stats for {len(instance_stats)} instances")
        print(f"  Git: {git_branch}@{git_commit[:8] if git_commit else 'N/A'}")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            run_id = insert_run(cur, run_spec, stats, suite, git_commit, git_branch)
            conn.commit()
            print(f"  Created eval_run id={run_id}")

            total = stream_and_store_samples(conn, cur, run_id, run_dir, instance_stats)

            cur.execute(
                "UPDATE eval_runs SET total_samples = %s, completed_at = NOW() WHERE id = %s",
                (total, run_id),
            )
            conn.commit()

            correct = sum(
                1 for s in instance_stats.values() if s.get("exact_match", 0) == 1.0
            )
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"  Done! run_id={run_id}, samples={total}, accuracy={accuracy:.1f}%")

        except Exception as e:
            conn.rollback()
            print(f"  Error: {e}")
            if "run_id" in locals():
                cur.execute("DELETE FROM eval_samples WHERE run_id = %s", (run_id,))
                cur.execute("DELETE FROM eval_runs WHERE id = %s", (run_id,))
                conn.commit()
                print(f"  Cleaned up partial run_id={run_id}")
            raise
        finally:
            cur.close()
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run HELM evaluation and store results in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fireworks model
  python helm_eval.py --model-name fireworks/kimi-k2p5 \\
    --api-base https://api.fireworks.ai/inference/v1 \\
    --api-model accounts/fireworks/models/kimi-k2p5 \\
    --benchmark aratrust --suite test-run --max-instances 10

  # Local model
  python helm_eval.py --model-name local/qwen3-5-9b \\
    --api-base http://127.0.0.1:1234/v1 --api-key lm-studio \\
    --benchmark arabic_mmlu --suite local-run --max-instances 10
        """,
    )

    parser.add_argument("--model-name", required=True, help="Model identifier (e.g. fireworks/kimi-k2p5)")
    parser.add_argument("--api-base", required=True, help="API base URL")
    parser.add_argument("--api-model", default=None, help="OpenAI model name for Fireworks (triggers FireworksNoThinkingClient)")
    parser.add_argument("--api-key", default=None, help="API key (default: from credentials.conf)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B", help="HuggingFace tokenizer (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--max-seq-len", type=int, default=131072, help="Max sequence length (default: 131072)")

    parser.add_argument("--benchmark", default="aratrust", choices=list(BENCHMARKS.keys()),
                        help="Benchmark to run (default: aratrust)")
    parser.add_argument("--benchmark-args", default=None, help="Override benchmark args (e.g. category=all)")
    parser.add_argument("--suite", required=True, help="Suite name for output dir and DB tracking")
    parser.add_argument("--max-instances", type=int, default=600, help="Max eval instances (default: 600)")

    parser.add_argument("--display-name", default=None, help="Display name for model metadata")
    parser.add_argument("--creator", default=None, help="Creator organization name")

    args = parser.parse_args()

    print("=== Configuring HELM ===")
    tokenizer_name = ensure_model_deployment(args)
    ensure_tokenizer_config(args, tokenizer_name)
    ensure_model_metadata(args)
    conf_path = generate_run_spec(args)
    update_credentials(args)

    try:
        run_helm(conf_path, args.suite, args.max_instances)
    finally:
        if conf_path.exists():
            conf_path.unlink()
            print(f"  Cleaned up {conf_path.name}")

    store_results(args.suite)

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
