#!/usr/bin/env python3
"""Unified CLI: configure HELM, run evaluation, and store results in DB — all in one command.

Usage:
    # Single benchmark
    python helm_eval.py \\
      --model-name fireworks/kimi-k2p5 \\
      --api-base https://api.fireworks.ai/inference/v1 \\
      --api-model accounts/fireworks/models/kimi-k2p5 \\
      --tokenizer Qwen/Qwen2.5-7B \\
      --benchmark aratrust \\
      --suite my-test \\
      --max-instances 10

    # Multiple benchmarks
    python helm_eval.py \\
      --model-name fireworks/kimi-k2p5 \\
      --api-base https://api.fireworks.ai/inference/v1 \\
      --api-model accounts/fireworks/models/kimi-k2p5 \\
      --tokenizer Qwen/Qwen2.5-7B \\
      --benchmark aratrust arabic_mmlu alghafa \\
      --suite my-test \\
      --max-instances 600
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from store_helm_results import (
    build_instance_stats_lookup,
    extract_category_from_run_name,
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
    "alrage": "",
    "madinah_qa": "subset=all",
}

# Generation benchmarks should NOT get the MCQ system prompt
GENERATION_BENCHMARKS = {"alrage"}

# Benchmarks that don't support "all" natively — must expand to individual subsets
ALGHAFA_SUBSETS = [
    "mcq_exams_test_ar",
    "meta_ar_dialects",
    "meta_ar_msa",
    "multiple_choice_facts_truefalse_balanced_task",
    "multiple_choice_grounded_statement_soqal_task",
    "multiple_choice_grounded_statement_xglue_mlqa_task",
    "multiple_choice_rating_sentiment_no_neutral_task",
    "multiple_choice_rating_sentiment_task",
    "multiple_choice_sentiment_task",
]

MADINAH_QA_SUBSETS = [
    "Arabic_Language_(General)",
    "Arabic_Language_(Grammar)",
]

ARABIC_EXAMS_SUBJECTS = [
    "Biology",
    "Islamic_Studies",
    "Physics",
    "Science",
    "Social",
]

ARABIC_MMLU_SUBSETS = [
    "Accounting_(University)",
    "Arabic_Language_(General)",
    "Arabic_Language_(Grammar)",
    "Arabic_Language_(High_School)",
    "Arabic_Language_(Middle_School)",
    "Arabic_Language_(Primary_School)",
    "Biology_(High_School)",
    "Civics_(High_School)",
    "Civics_(Middle_School)",
    "Computer_Science_(High_School)",
    "Computer_Science_(Middle_School)",
    "Computer_Science_(Primary_School)",
    "Computer_Science_(University)",
    "Driving_Test",
    "Economics_(High_School)",
    "Economics_(Middle_School)",
    "Economics_(University)",
    "General_Knowledge",
    "General_Knowledge_(Middle_School)",
    "General_Knowledge_(Primary_School)",
    "Geography_(High_School)",
    "Geography_(Middle_School)",
    "Geography_(Primary_School)",
    "History_(High_School)",
    "History_(Middle_School)",
    "History_(Primary_School)",
    "Islamic_Studies",
    "Islamic_Studies_(High_School)",
    "Islamic_Studies_(Middle_School)",
    "Islamic_Studies_(Primary_School)",
    "Law_(Professional)",
    "Management_(University)",
    "Math_(Primary_School)",
    "Natural_Science_(Middle_School)",
    "Natural_Science_(Primary_School)",
    "Philosophy_(High_School)",
    "Physics_(High_School)",
    "Political_Science_(University)",
    "Social_Science_(Middle_School)",
    "Social_Science_(Primary_School)",
]

ARABIC_MMMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


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
        fireworks_args = {
            "base_url": args.api_base,
            "openai_model_name": args.api_model,
        }
        if args._current_benchmark in GENERATION_BENCHMARKS:
            fireworks_args["system_prompt"] = ""
        client_spec = {
            "class_name": "fireworks_client.FireworksNoThinkingClient",
            "args": fireworks_args,
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


def _fetch_tokenizer_tokens(tokenizer_id):
    """Fetch EOS/BOS tokens from HuggingFace tokenizer_config.json."""
    import json
    import urllib.request

    url = f"https://huggingface.co/{tokenizer_id}/raw/main/tokenizer_config.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            config = json.loads(resp.read())

        eos = config.get("eos_token")
        bos = config.get("bos_token")

        # Some configs store tokens as dicts with "content" key
        if isinstance(eos, dict):
            eos = eos.get("content", eos)
        if isinstance(bos, dict):
            bos = bos.get("content", bos)

        return eos, bos
    except Exception as e:
        print(f"  Warning: Could not fetch tokenizer config for {tokenizer_id}: {e}")
        print("  Falling back to default tokens (<|im_end|>, <|im_start|>)")
        return "<|im_end|>", "<|im_start|>"


def ensure_tokenizer_config(args, tokenizer_name):
    """Add or update tokenizer entry in tokenizer_configs.yaml."""
    path = PROJECT_DIR / "tokenizer_configs.yaml"
    print("Path: ", path)
    data = load_yaml(path)
    configs = data.setdefault("tokenizer_configs", [])

    eos_token, bos_token = _fetch_tokenizer_tokens(args.tokenizer)
    print(f"  Tokenizer tokens: eos={eos_token!r}, bos={bos_token!r}")

    entry = {
        "name": tokenizer_name,
        "tokenizer_spec": {
            "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
            "args": {"pretrained_model_name_or_path": args.tokenizer, "trust_remote_code": True},
        },
        "end_of_text_token": eos_token,
        "prefix_token": bos_token,
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


def _expand_benchmark_entries(benchmark, bench_args, model_name):
    """Expand a benchmark into one or more run spec entries.

    Benchmarks like alghafa and arabic_mmmlu don't support 'all' natively,
    so we expand them into individual subset entries.
    """
    if benchmark == "madinah_qa" and bench_args == "subset=all":
        return [f"madinah_qa:subset={s},model={model_name}" for s in MADINAH_QA_SUBSETS]
    if benchmark == "arabic_mmlu" and bench_args == "subset=all":
        return [f"arabic_mmlu:subset={s},model={model_name}" for s in ARABIC_MMLU_SUBSETS]
    if benchmark == "arabic_exams" and bench_args == "subject=all":
        return [f"arabic_exams:subject={s},model={model_name}" for s in ARABIC_EXAMS_SUBJECTS]
    if benchmark == "alghafa" and bench_args == "subset=all":
        return [f"alghafa:subset={s},model={model_name}" for s in ALGHAFA_SUBSETS]
    if benchmark == "arabic_mmmlu" and bench_args == "subject=all":
        return [f"mbzuai_human_translated_arabic_mmlu:subject={s},model={model_name}" for s in ARABIC_MMMLU_SUBJECTS]
    # arabic_mmmlu uses a different HELM run spec name
    if benchmark == "arabic_mmmlu":
        return [f"mbzuai_human_translated_arabic_mmlu:{bench_args},model={model_name}"]
    if bench_args:
        return [f"{benchmark}:{bench_args},model={model_name}"]
    return [f"{benchmark}:model={model_name}"]


def generate_run_spec(args):
    """Generate a run_specs conf file for this run."""
    benchmark = args._current_benchmark
    bench_args = args.benchmark_args or BENCHMARKS.get(benchmark, "")
    entries = _expand_benchmark_entries(benchmark, bench_args, args.model_name)

    lines = ",\n".join(f'  {{description: "{e}", priority: 1}}' for e in entries)
    conf_path = PROJECT_DIR / f"run_specs_{args.suite}.conf"
    conf_path.write_text(f"entries: [\n{lines}\n]\n", encoding="utf-8")
    print(f"  {conf_path.name}: {len(entries)} entries ({benchmark})")
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


def run_helm(conf_path, suite, max_instances, num_threads=1):
    """Run helm-run as a subprocess."""
    cmd = [
        "helm-run",
        "--conf-paths", str(conf_path),
        "--suite", suite,
        "--local-path", ".",
        "--max-eval-instances", str(max_instances),
        "-n", str(num_threads),
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
    """Find run directories and store results in DB.

    Multiple run dirs that share the same benchmark (e.g. alghafa subsets)
    are grouped into a single eval_run with subset stored in eval_samples.category.
    """
    import json as _json

    print(f"\n=== Storing Results ===")
    suite_dir = PROJECT_DIR / "benchmark_output" / "runs" / suite

    if not suite_dir.exists():
        print(f"Error: suite directory not found: {suite_dir}")
        sys.exit(1)

    run_dirs = sorted(
        [d for d in suite_dir.iterdir() if d.is_dir() and (d / "scenario_state.json").exists()]
    )
    if not run_dirs:
        print(f"Error: no completed runs found in {suite_dir}")
        sys.exit(1)

    git_commit, git_branch = get_git_info()

    # Group run dirs by base benchmark name (before the colon)
    groups = {}
    for run_dir in run_dirs:
        base_name = run_dir.name.split(":")[0]  # e.g. "alghafa"
        groups.setdefault(base_name, []).append(run_dir)

    for base_name, dirs in groups.items():
        # Use first run dir to create the parent eval_run
        first_dir = dirs[0]
        required = ("run_spec.json", "scenario_state.json", "per_instance_stats.json", "stats.json")
        missing = [f for f in required if not (first_dir / f).exists()]
        if missing:
            print(f"  Skipping {base_name} — missing: {', '.join(missing)}")
            continue

        run_spec = load_json(first_dir / "run_spec.json")
        stats = load_json(first_dir / "stats.json")

        print(f"\nStoring: {base_name} ({len(dirs)} subset(s))")
        print(f"  Git: {git_branch}@{git_commit[:8] if git_commit else 'N/A'}")

        conn = get_db_connection()
        cur = conn.cursor()
        run_id = None

        try:
            run_id = insert_run(cur, run_spec, stats, suite, git_commit, git_branch)
            conn.commit()
            print(f"  Created eval_run id={run_id}")

            grand_total = 0
            total_correct = 0
            total_score_sum = 0.0
            is_generation = False
            per_subset_metrics = {}

            for run_dir in dirs:
                req = [f for f in required if not (run_dir / f).exists()]
                if req:
                    print(f"  Skipping {run_dir.name} — missing: {', '.join(req)}")
                    continue

                category = extract_category_from_run_name(run_dir.name)
                subset_label = category or run_dir.name

                per_instance_stats = load_json(run_dir / "per_instance_stats.json")
                instance_stats = build_instance_stats_lookup(per_instance_stats)

                total = stream_and_store_samples(conn, cur, run_id, run_dir, instance_stats, category=category)
                grand_total += total

                # Accumulate correct/scores per subset (avoids instance_id collision)
                if any("alrage_score" in s for s in instance_stats.values()):
                    is_generation = True
                    total_score_sum += sum(s.get("alrage_score", 0) for s in instance_stats.values())
                else:
                    total_correct += sum(1 for s in instance_stats.values() if s.get("exact_match", 0) == 1.0)

                # Per-subset accuracy from stats.json
                subset_stats = load_json(run_dir / "stats.json")
                for stat in subset_stats:
                    name = stat["name"]["name"]
                    if name in ("exact_match", "alrage_score"):
                        per_subset_metrics[subset_label] = stat.get("mean", stat.get("sum"))

                print(f"  {subset_label}: {total} samples")

            # Update run with aggregated totals and per-subset metrics
            if is_generation:
                overall = total_score_sum / grand_total if grand_total > 0 else 0
                agg_metrics = {"alrage_score": overall, "per_subset": per_subset_metrics}
            else:
                overall = total_correct / grand_total if grand_total > 0 else 0
                agg_metrics = {"exact_match": overall, "per_subset": per_subset_metrics}

            cur.execute(
                "UPDATE eval_runs SET total_samples = %s, metrics = %s, completed_at = NOW() WHERE id = %s",
                (grand_total, _json.dumps(agg_metrics, ensure_ascii=False), run_id),
            )
            conn.commit()

            if is_generation:
                print(f"  Done! run_id={run_id}, samples={grand_total}, avg_score={overall:.3f}")
            else:
                print(f"  Done! run_id={run_id}, samples={grand_total}, accuracy={overall*100:.1f}%")

        except Exception as e:
            conn.rollback()
            print(f"  Error: {e}")
            if run_id is not None:
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
  # Single benchmark
  python helm_eval.py --model-name fireworks/kimi-k2p5 \\
    --api-base https://api.fireworks.ai/inference/v1 \\
    --api-model accounts/fireworks/models/kimi-k2p5 \\
    --benchmark aratrust --suite test-run --max-instances 10

  # Multiple benchmarks
  python helm_eval.py --model-name fireworks/kimi-k2p5 \\
    --api-base https://api.fireworks.ai/inference/v1 \\
    --api-model accounts/fireworks/models/kimi-k2p5 \\
    --benchmark aratrust arabic_mmlu alghafa --suite full-run --max-instances 600

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

    parser.add_argument("--benchmark", nargs="+", default=["aratrust"],
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmark(s) to run (default: aratrust)")
    parser.add_argument("--benchmark-args", default=None, help="Override benchmark args (e.g. category=all)")
    parser.add_argument("--suite", required=True, help="Suite name for output dir and DB tracking")
    parser.add_argument("--max-instances", type=int, default=99999, help="Max eval instances per subset (default: no practical limit)")

    parser.add_argument("-n", "--num-threads", type=int, default=1, help="Number of parallel threads (default: 1)")

    parser.add_argument("--display-name", default=None, help="Display name for model metadata")
    parser.add_argument("--creator", default=None, help="Creator organization name")

    args = parser.parse_args()

    print("=== Configuring HELM ===")
    # One-time setup (not benchmark-dependent)
    args._current_benchmark = args.benchmark[0]
    tokenizer_name = ensure_model_deployment(args)
    ensure_tokenizer_config(args, tokenizer_name)
    ensure_model_metadata(args)
    update_credentials(args)

    benchmarks = args.benchmark
    print(f"\nBenchmarks to run: {', '.join(benchmarks)}")

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n=== Benchmark {i}/{len(benchmarks)}: {benchmark} ===")
        args._current_benchmark = benchmark

        # Reconfigure model deployment for correct system prompt (MCQ vs generation)
        ensure_model_deployment(args)

        conf_path = generate_run_spec(args)
        try:
            run_helm(conf_path, args.suite, args.max_instances, args.num_threads)
        finally:
            if conf_path.exists():
                conf_path.unlink()
                print(f"  Cleaned up {conf_path.name}")

    store_results(args.suite)

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
