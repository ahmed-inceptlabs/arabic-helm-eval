#!/usr/bin/env python3
"""Generate synthetic Arabic grammar data (MCQ + dialogue) via OpenAI-compatible API."""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
from pydantic import BaseModel

from models import MCQResponse, DialogueResponse


SCRIPT_DIR = Path(__file__).parent
PROMPT_PATH = SCRIPT_DIR / "prompts" / "madinah_curriculum.yaml"
RAW_DIR = SCRIPT_DIR / "raw"
CREDS_PATH = SCRIPT_DIR.parent / "credentials.conf"

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_curriculum(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_key_from_credentials(api_base: str) -> Optional[str]:
    if not CREDS_PATH.exists():
        return None
    key_name = "openaiApiKey"
    if "fireworks.ai" in api_base:
        key_name = "fireworksApiKey"
    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith(f"{key_name}:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


# --- Fallback functions for --no-structured-output mode ---

def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _validate_mcq(payload: dict):
    question = payload.get("question", "").strip()
    options = payload.get("options", {})
    answer = payload.get("answer", "").strip()
    if not question or not isinstance(options, dict):
        raise ValueError("Missing question or options")
    for key in ["أ", "ب", "ج", "د"]:
        if key not in options or not str(options[key]).strip():
            raise ValueError("Incomplete options set")
    if answer not in options:
        raise ValueError("Answer not in options")
    return question, options, answer


def _validate_dialogue(payload: dict):
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Dialogue must contain >= 2 messages")
    for m in messages:
        if m.get("role") not in {"user", "assistant"}:
            raise ValueError("Dialogue roles must be user/assistant")
        if not m.get("content"):
            raise ValueError("Dialogue message content is empty")
    return messages

# --- End fallback functions ---


def _render_template(template: str, topic: dict) -> str:
    focus = topic.get("focus", [])
    focus_str = "، ".join(focus) if isinstance(focus, list) else str(focus)
    return (
        template
        .replace("{topic}", topic["name"])
        .replace("{difficulty}", topic["difficulty"])
        .replace("{focus}", focus_str)
    )


def _make_messages(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_model(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float,
    max_retries: int,
    response_model: type[BaseModel] | None = None,
):
    for attempt in range(max_retries):
        try:
            if response_model is not None:
                resp = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_model,
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model returned refusal or unparseable response")
                return parsed
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"API error: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


def generate_examples(
    client: OpenAI,
    model: str,
    curriculum: dict,
    max_examples: int,
    mcq_fraction: float,
    temperature: float,
    max_retries: int,
    use_structured: bool = True,
):
    topics = curriculum.get("topics", [])
    templates = curriculum.get("templates", {})
    mcq_template = templates.get("mcq_prompt", "")
    dialogue_template = templates.get("dialogue_prompt", "")
    system_prompt = curriculum.get(
        "system_prompt",
        "أنت نموذج مساعد يلتزم بإخراج JSON فقط دون أي نص إضافي.",
    )

    if not topics or not mcq_template:
        raise ValueError("Curriculum topics or MCQ template missing")

    mcq_target = int(max_examples * mcq_fraction)
    dialogue_target = max_examples - mcq_target

    topic_cycle = topics[:]
    random.shuffle(topic_cycle)
    topic_idx = 0

    def next_topic():
        nonlocal topic_idx, topic_cycle
        if topic_idx >= len(topic_cycle):
            topic_cycle = topics[:]
            random.shuffle(topic_cycle)
            topic_idx = 0
        topic = topic_cycle[topic_idx]
        topic_idx += 1
        return topic

    results = []

    # --- MCQ generation ---
    mcq_done = 0
    mcq_attempts = 0
    mcq_max_attempts = max(mcq_target * 3, 1)
    while mcq_done < mcq_target and mcq_attempts < mcq_max_attempts:
        mcq_attempts += 1
        topic = next_topic()
        prompt = _render_template(mcq_template, topic)
        messages = _make_messages(system_prompt, prompt)
        try:
            if use_structured:
                mcq = _call_model(
                    client, model, messages, temperature, max_retries,
                    response_model=MCQResponse,
                )
                question = mcq.question
                options = mcq.options
                answer = mcq.answer
                raw = mcq.model_dump_json(ensure_ascii=False)
            else:
                raw = _call_model(client, model, messages, temperature, max_retries)
                payload = _extract_json(raw)
                question, options, answer = _validate_mcq(payload)
        except Exception as e:
            print(f"MCQ generation error: {e}", file=sys.stderr)
            continue
        user_text = (
            f"{question}\n"
            f"أ) {options['أ']}\n"
            f"ب) {options['ب']}\n"
            f"ج) {options['ج']}\n"
            f"د) {options['د']}"
        )
        results.append(
            {
                "source": "synthetic_madinah",
                "type": "mcq",
                "topic": topic["name"],
                "difficulty": topic["difficulty"],
                "question": question,
                "options": options,
                "answer": answer,
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": answer},
                ],
                "raw_prompt": prompt,
                "raw_response": raw,
            }
        )
        mcq_done += 1

    # --- Dialogue generation ---
    dialogue_done = 0
    dialogue_attempts = 0
    dialogue_max_attempts = max(dialogue_target * 3, 1)
    while dialogue_done < dialogue_target and dialogue_attempts < dialogue_max_attempts:
        dialogue_attempts += 1
        if not dialogue_template:
            break
        topic = next_topic()
        prompt = _render_template(dialogue_template, topic)
        messages = _make_messages(system_prompt, prompt)
        try:
            if use_structured:
                dialogue = _call_model(
                    client, model, messages, temperature, max_retries,
                    response_model=DialogueResponse,
                )
                messages_out = [m.model_dump() for m in dialogue.messages]
                raw = dialogue.model_dump_json(ensure_ascii=False)
            else:
                raw = _call_model(client, model, messages, temperature, max_retries)
                payload = _extract_json(raw)
                messages_out = _validate_dialogue(payload)
        except Exception as e:
            print(f"Dialogue generation error: {e}", file=sys.stderr)
            continue
        results.append(
            {
                "source": "synthetic_madinah",
                "type": "dialogue",
                "topic": topic["name"],
                "difficulty": topic["difficulty"],
                "messages": messages_out,
                "raw_prompt": prompt,
                "raw_response": raw,
            }
        )
        dialogue_done += 1

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--mcq-fraction", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-config", default=str(PROMPT_PATH))
    parser.add_argument("--out", default=str(RAW_DIR / "synthetic_grammar.jsonl"))
    parser.add_argument(
        "--no-structured-output", action="store_true",
        help="Disable structured output; fall back to regex JSON extraction",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    curriculum = _load_curriculum(Path(args.prompt_config))

    api_key = args.api_key.strip()
    if not api_key:
        api_key = _read_key_from_credentials(args.api_base) or ""
    if not api_key:
        raise RuntimeError("API key not provided and not found in credentials.conf")

    client = OpenAI(api_key=api_key, base_url=args.api_base)
    rows = generate_examples(
        client,
        args.model,
        curriculum,
        args.max_examples,
        args.mcq_fraction,
        args.temperature,
        args.max_retries,
        use_structured=not args.no_structured_output,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows):,} synthetic rows to {out_path}")


if __name__ == "__main__":
    main()
