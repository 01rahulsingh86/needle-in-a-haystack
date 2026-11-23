
import csv
import random
import re
import time
import requests

MODEL = "qwen3:14b"
URL = "http://localhost:11434/api/generate"

CONTEXT_LENGTHS = [2000, 4000, 8000]
DEPTHS = [0, 25, 50, 75, 100]

NEEDLE = "The secret launch code for Project Orion is BLUE-7492."
ANSWER = "BLUE-7492"

FILLER = (
    "The research team reviewed project documentation, "
    "software architecture, operational requirements, "
    "testing processes, and engineering reports. "
)

def make_prompt(target_tokens, depth):
    # Rough token approximation, not an exact token count.
    target_chars = target_tokens * 4
    rng = random.Random(42)
    sentences = [FILLER] * (target_chars // len(FILLER) + 1)
    rng.shuffle(sentences)
    haystack = "".join(sentences)[:target_chars]

    position = int(len(haystack) * depth / 100)
    content = (
        haystack[:position]
        + "\n" + NEEDLE + "\n"
        + haystack[position:]
    )

    return (
        "Read the following document carefully.\n"
        "Only use the document to answer.\n"
        "Return only the launch code, without explanation.\n"
        "<document>\n"
        + content
        + "\n</document>\n"
        "Question: What is the secret launch code "
        "for Project Orion?"
    )

def evaluate(target_tokens, depth):
    prompt = make_prompt(target_tokens, depth)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": max(8192, target_tokens + 4096),
            "num_predict": 100,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        URL, json=payload, timeout=600
    )
    response.raise_for_status()
    data = response.json()

    elapsed = time.perf_counter() - start
    answer = data.get("response", "").strip()

    # Strict accuracy: answer must be exactly the code.
    correct = answer == ANSWER

    return {
        "target_tokens_approx": target_tokens,
        "depth_pct": depth,
        "actual_prompt_tokens": data.get("prompt_eval_count"),
        "actual_answer": answer,
        "correct": int(correct),
        "latency_sec": round(elapsed, 2),
    }

def main():
    with open("results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_tokens_approx",
                "depth_pct",
                "actual_prompt_tokens",
                "actual_answer",
                "correct",
                "latency_sec",
            ],
        )
        writer.writeheader()

        for length in CONTEXT_LENGTHS:
            for depth in DEPTHS:
                print(f"Testing {length} tokens, depth {depth}%")

                try:
                    result = evaluate(length, depth)
                    writer.writerow(result)
                    file.flush()

                    status = "PASS" if result["correct"] else "FAIL"
                    print(
                        f"{status} | "
                        f"{result['latency_sec']}s | "
                        f"{result['actual_answer'][:80]}"
                    )
                except Exception as error:
                    print(f"ERROR: {error}")

    print("\nFinished. Results saved to results.csv")

if __name__ == "__main__":
    main()