
import csv
import random
import re
import secrets
import time
import requests
from pathlib import Path

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================

MODEL = "qwen3:14b"
URL = "http://localhost:11434/api/generate"

DEPTHS = [0, 50, 100]
TRIALS = 1

NUM_CTX = 24576
OUTPUT_FILE = Path("results_v2_diagnostic.csv")

FILLER = (
    "The engineering team documented a new software release. "
    "Scientists studied the relationship between data and prediction. "
    "A university laboratory evaluated its research methods. "
    "The system recorded performance and reliability measurements. "
    "Researchers investigated unexpected experimental results. "
)

# Short document to test potential truncation.
HAYSTACK = FILLER * 80


# ==========================================
# 2. GENERATE THE TEST CASE
# ==========================================

def create_test(depth, trial):
    code = f"ATLAS-{secrets.randbelow(9000) + 1000}"

    needle = (
        "The secret verification code for Project Atlas "
        f"is {code}."
    )

    position = int(len(HAYSTACK) * depth / 100)

    document = (
        HAYSTACK[:position]
        + "\n"
        + needle
        + "\n"
        + HAYSTACK[position:]
    )

    prompt = (
        "Read the document carefully.\n"
        "Return only the secret verification code "
        "for Project Atlas.\n\n"
        "DOCUMENT:\n"
        + document
        + "\n\nQUESTION:\n"
        "What is the secret verification code for Project Atlas?"
    )

    # Verify the needle is present exactly once.
    assert prompt.count(needle) == 1, (
        "Needle missing or duplicated!"
    )

    actual_position_pct = (
        prompt.index(needle) / len(prompt) * 100
    )

    return prompt, code, actual_position_pct


# ==========================================
# 3. RUN ONE EVALUATION
# ==========================================

def run_test(depth, trial):
    prompt, expected, position_pct = create_test(
        depth, trial
    )

    print(f"Prompt characters: {len(prompt):,}")
    print(f"Needle position: {position_pct:.2f}%")

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": 100,
            "temperature": 0,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()
    data = response.json()

    latency = time.perf_counter() - start

    answer = data.get("response", "").strip()

    # ======================================
    # 4. SCORING
    # ======================================

    detected_codes = re.findall(
        r"\bATLAS-\d{4}\b",
        answer,
    )

    retrieval_correct = detected_codes == [expected]
    exact_match = answer == expected

    # ======================================
    # 5. TOKEN AND TRUNCATION DIAGNOSTICS
    # ======================================

    prompt_tokens = data.get("prompt_eval_count")

    print("Reported prompt tokens:", prompt_tokens)

    # A heuristic warning, NOT proof of truncation.
    half_context = NUM_CTX // 2

    if (
        prompt_tokens is not None
        and abs(prompt_tokens - half_context) <= 10
    ):
        print(
            "WARNING: Prompt token count is close "
            "to half of num_ctx."
        )
        print(
            "This warrants investigation for "
            "possible truncation or context handling."
        )

    print("Expected:", expected)
    print("Actual:", answer)
    print("Retrieval correct:", retrieval_correct)
    print("Exact match:", exact_match)

    return {
        "model": MODEL,
        "depth": depth,
        "trial": trial,
        "needle_position_pct": round(position_pct, 2),
        "prompt_characters": len(prompt),
        "expected": expected,
        "response": answer,
        "retrieval_correct": int(retrieval_correct),
        "exact_match": int(exact_match),
        "prompt_tokens": prompt_tokens,
        "num_ctx": NUM_CTX,
        "latency_sec": round(latency, 2),
        "prompt_eval_sec": round(
            data.get("prompt_eval_duration", 0) / 1e9,
            2,
        ),
        "generation_sec": round(
            data.get("eval_duration", 0) / 1e9,
            2,
        ),
        "load_sec": round(
            data.get("load_duration", 0) / 1e9,
            2,
        ),
        "done_reason": data.get("done_reason", ""),
    }


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(rows):
    if not rows:
        return

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 7. EXPERIMENT RUNNER
# ==========================================

def main():
    configurations = [
        (depth, trial)
        for depth in DEPTHS
        for trial in range(1, TRIALS + 1)
    ]

    random.Random(42).shuffle(configurations)

    results = []

    print("=" * 60)
    print("LLM CONTEXT RETRIEVAL — DIAGNOSTIC BENCHMARK")
    print("=" * 60)

    print("Model:", MODEL)
    print("Configured context:", NUM_CTX)
    print("Document characters:", len(HAYSTACK))
    print("Total tests:", len(configurations))
    print()

    for depth, trial in configurations:
        print("\n" + "=" * 60)
        print(f"Depth: {depth}% | Trial: {trial}")
        print("=" * 60)

        try:
            result = run_test(depth, trial)

            results.append(result)
            save_results(results)

            status = (
                "PASS"
                if result["retrieval_correct"]
                else "FAIL"
            )

            print(
                f"\n{status} | "
                f"{result['prompt_tokens']} tokens | "
                f"{result['latency_sec']}s"
            )

        except Exception as exc:
            print("ERROR:", exc)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for depth in DEPTHS:
        subset = [
            r for r in results
            if r["depth"] == depth
        ]

        if subset:
            correct = sum(
                r["retrieval_correct"]
                for r in subset
            )

            accuracy = correct / len(subset)

            print(
                f"Depth {depth}%: "
                f"{correct}/{len(subset)} passed "
                f"({accuracy:.0%})"
            )

    print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
