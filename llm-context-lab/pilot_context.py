
import requests
import time
import re


MODEL = "qwen3:14b"

# Long, varied text generated from a fixed set of sentences.
paragraphs = [
    "The engineering team documented a new software release.",
    "Scientists studied the relationship between data and prediction.",
    "A university laboratory evaluated its research methods.",
    "The system recorded performance and reliability measurements.",
    "Researchers investigated unexpected experimental results.",
]

text = " ".join(paragraphs * 1300)

needle = (
    "The secret verification code for Project Atlas "
    "is ATLAS-9274."
)

midpoint = 0
document = text[:midpoint] + "\n" + needle + "\n" + text[midpoint:]

prompt = f"""
Read the document carefully.
Return only the secret verification code for Project Atlas.

DOCUMENT:
{document}

QUESTION:
What is the secret verification code for Project Atlas?
"""

assert prompt.count(needle) == 1, "Needle missing or duplicated"

print("Needle found:", needle in prompt)
print("Needle character position:", prompt.index(needle))
print("Total prompt characters:", len(prompt))

print("Running pilot retrieval test...")
print("Prompt characters:", len(prompt))

start = time.perf_counter()

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 24576,
            "num_predict": 100,
            "temperature": 0,
        },
    },
    timeout=600,
)

response.raise_for_status()
data = response.json()

answer = data.get("response", "").strip()
elapsed = time.perf_counter() - start

print("Answer:", answer)
expected = "ATLAS-9274"

# Strict exact match
exact_match = answer == expected

# Extract codes matching the expected format
codes = re.findall(r"\bATLAS-\d{4}\b", answer)

# Pass only if the response contains exactly one code,
# and that code is the expected answer
retrieval_correct = codes == [expected]

# Check whether the model followed the output instruction
format_correct = exact_match

print("Exact Match:", exact_match)
print("Retrieval Correct:", retrieval_correct)
print("Format Correct:", format_correct)
print("Detected Codes:", codes)
print("Actual prompt tokens:", data.get("prompt_eval_count"))
print("Elapsed seconds:", round(elapsed, 2))
