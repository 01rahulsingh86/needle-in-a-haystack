"""Token-budgeted needle-in-a-haystack retrieval benchmark for Ollama/Qwen3.

Usage:
  python benchmark_v3.py --dry-run  # build/validate prompts only
  python benchmark_v3.py            # run 9 pilot evaluations

Requires: pip install requests transformers tokenizers
"""
import argparse
import csv
import random
import re
import secrets
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL = "qwen3:14b"
TOKENIZER = "Qwen/Qwen3-14B"
API_URL = "http://127.0.0.1:11434/api/generate"
NUM_CTX = 40960
NUM_PREDICT = 80
# We independently count tokens and reserve ample input/output headroom.
MAX_ALLOWED_INPUT_TOKENS = 26000
TARGETS = [24000]
DEPTHS = [0, 50, 100]
TRIALS = 1  # Pilot: one test at each depth. Increase to 3 after validation.
OUTPUT = Path("results_24k.csv")

INTRO = (
    "Read the following collection of research notes. "
    "Find the secret verification code for Project Atlas. "
    "Return ONLY the code, in the format ATLAS-1234.\n\n"
    "<document>\n"
)
OUTRO = (
    "\n</document>\n\n"
    "Question: What is the secret verification code for Project Atlas?"
)
TOPICS = [
    "robot motion planning", "language translation", "image segmentation",
    "water conservation", "crop disease monitoring", "software testing",
    "urban transport", "ocean temperature", "wildlife migration",
    "renewable energy", "materials engineering", "forest monitoring",
    "medical image annotation", "satellite imagery", "traffic modeling",
    "speech recognition", "computer graphics", "weather prediction",
]
METHODS = [
    "a controlled observation", "an annotated dataset", "a simulation study",
    "a prototype demonstration", "a calibration experiment",
    "a comparative assessment", "a field measurement",
]
OUTCOMES = [
    "measurement noise required additional validation",
    "the sample size was insufficient for a firm conclusion",
    "the second trial supported the initial observation",
    "independent reviewers questioned the sampling procedure",
    "several missing values were excluded from the summary",
    "the team scheduled a follow-up measurement",
    "the error bars were wider than anticipated",
]


def make_paragraph(index, rng):
    # Distinct records with varied topics and wording, rather than one repeated sentence.
    topic = rng.choice(TOPICS)
    method = rng.choice(METHODS)
    outcome = rng.choice(OUTCOMES)
    group = rng.choice(["North", "South", "West", "East"])
    month = rng.choice(["January", "March", "May", "July", "September", "November"])
    return (
        f"Record {index:05d}: During {month}, the {group} research group studied "
        f"{topic} using {method}. Review notes indicated that {outcome}. "
        f"The investigators archived observations under batch {rng.randrange(100000,999999)} "
        f"and discussed possible follow-up procedures.\n"
    )


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def create_test(tokenizer, target, depth, trial):
    rng = random.Random(100000 * target + 1000 * trial)
    code = f"ATLAS-{secrets.randbelow(9000) + 1000}"
    needle = f"The secret verification code for Project Atlas is {code}."

    # Build a distinct-document haystack close to the requested token budget.
    # Budget for the wrapper and needle, plus margin for token-boundary effects.
    reserved = count_tokens(tokenizer, INTRO + needle + OUTRO) + 20
    filler_budget = target - reserved
    if filler_budget <= 0:
        raise ValueError("Target is too small for prompt wrapper")

    records = []
    running = 0
    idx = 1
    while running < filler_budget:
        record = make_paragraph(idx, rng)
        n = count_tokens(tokenizer, record)
        if running + n > filler_budget:
            break
        records.append((record, n))
        running += n
        idx += 1

    # Put the needle at the specified *approximate token depth*, at a record boundary.
    cumulative = 0
    insert_at = len(records)
    if depth == 0:
        insert_at = 0
    elif depth != 100:
        for j, (_, n) in enumerate(records):
            if cumulative + n >= running * depth / 100:
                insert_at = j + 1
                break
            cumulative += n

    before = "".join(record for record, _ in records[:insert_at])
    after = "".join(record for record, _ in records[insert_at:])
    prompt = INTRO + before + needle + "\n" + after + OUTRO
    exact_tokens = count_tokens(tokenizer, prompt)
    prefix_tokens = count_tokens(tokenizer, INTRO + before)
    actual_depth = 100 * prefix_tokens / exact_tokens

    if exact_tokens > MAX_ALLOWED_INPUT_TOKENS:
        raise ValueError(
            f"Full prompt has {exact_tokens} tokens, above safety ceiling "
            f"{MAX_ALLOWED_INPUT_TOKENS}. No API call was made."
        )
    if exact_tokens + NUM_PREDICT + 128 > NUM_CTX:
        raise ValueError("Insufficient configured context for prompt + response")
    if prompt.count(needle) != 1:
        raise AssertionError("Needle missing or duplicated")

    return prompt, code, exact_tokens, round(actual_depth, 2)


def evaluate(session, prompt, expected, estimated_tokens):
    started = time.perf_counter()
    response = session.post(
        API_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
                "temperature": 0,
            },
        },
        timeout=1200,
    )
    response.raise_for_status()
    data = response.json()
    wall_time = time.perf_counter() - started
    answer = data.get("response", "").strip()
    reported = data.get("prompt_eval_count")

    # Compare independently counted tokenizer tokens with Ollama's processing count.
    # A small mismatch is expected because of model template and tokenizer differences.
    tolerance = max(150, int(estimated_tokens * 0.10))
    valid_input = (
        isinstance(reported, int)
        and abs(reported - estimated_tokens) <= tolerance
    )

    found_codes = re.findall(r"\bATLAS-\d{4}\b", answer)
    return {
        "expected": expected,
        "answer": answer,
        "tokens_hf": estimated_tokens,
        "tokens_ollama": reported,
        "input_valid": int(valid_input),
        "retrieval_correct": int(found_codes == [expected]) if valid_input else "",
        "format_correct": int(answer == expected) if valid_input else "",
        "elapsed_sec": round(wall_time, 2),
        "prompt_eval_sec": round(data.get("prompt_eval_duration", 0) / 1e9, 2),
        "generation_sec": round(data.get("eval_duration", 0) / 1e9, 2),
        "load_sec": round(data.get("load_duration", 0) / 1e9, 2),
        "done_reason": data.get("done_reason", ""),
    }


def save(rows):
    if not rows:
        return
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer (full model weights are not downloaded)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    cases = [(t, d, n) for t in TARGETS for d in DEPTHS for n in range(1, TRIALS + 1)]
    random.Random(42).shuffle(cases)
    results = []

    print(f"Configured context: {NUM_CTX}; maximum input token ceiling: {MAX_ALLOWED_INPUT_TOKENS}")
    print(f"Cases: {len(cases)} | dry-run: {args.dry_run}")
    with requests.Session() as session:
        for target, depth, trial in cases:
            prompt, code, tokens, real_depth = create_test(tokenizer, target, depth, trial)
            print(f"Target {target:>5} | depth {depth:>3}% | trial {trial} "
                  f"| counted {tokens:>5} tokens | actual depth {real_depth:.1f}%", flush=True)
            if args.dry_run:
                continue

            try:
                result = evaluate(session, prompt, code, tokens)
            except requests.RequestException as exc:
                print(f"  REQUEST ERROR: {exc}; previous results preserved")
                break
            row = {"target_tokens": target, "requested_depth": depth,
                   "actual_depth": real_depth, "trial": trial, **result}
            results.append(row)
            save(results)

            if not result["input_valid"]:
                print(f"  INVALID INPUT: expected ~{tokens}, Ollama reported "
                      f"{result['tokens_ollama']}. Stopping to avoid misleading scores.")
                break
            status = "PASS" if result["retrieval_correct"] else "FAIL"
            print(f"  {status} | Ollama tokens {result['tokens_ollama']} "
                  f"| {result['elapsed_sec']}s | response: {result['answer'][:90]!r}", flush=True)

    if args.dry_run:
        print("Dry run passed: no model requests were sent.")
        return
    print(f"\nSaved {len(results)} results in {OUTPUT}")
    valid = [r for r in results if r["input_valid"]]
    if valid:
        print(f"Valid test accuracy: {sum(r['retrieval_correct'] for r in valid)}/{len(valid)}")
    else:
        print("No valid test results; investigate input token mismatch.")


if __name__ == "__main__":
    main()
