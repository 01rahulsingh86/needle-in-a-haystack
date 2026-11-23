
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-14B"

FILLER = (
    "The engineering team documented a new software release. "
    "Scientists studied the relationship between data and prediction. "
    "A university laboratory evaluated its research methods. "
    "The system recorded performance and reliability measurements. "
    "Researchers investigated unexpected experimental results. "
)

# Rebuild the ORIGINAL long test
haystack = FILLER * 1300

needle = (
    "The secret verification code for Project Atlas "
    "is ATLAS-9274."
)

midpoint = len(haystack) // 2

document = (
    haystack[:midpoint]
    + "\n"
    + needle
    + "\n"
    + haystack[midpoint:]
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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokens = tokenizer.encode(
    prompt,
    add_special_tokens=False
)

print("\n===== TOKEN DIAGNOSTIC =====")
print("Prompt characters:", len(prompt))
print("Full prompt tokens:", len(tokens))
print("Configured Ollama context:", 24576)
print("Previously reported tokens:", 12290)
print("Half configured context:", 24576 // 2)
print("Needle present:", needle in prompt)

if len(tokens) > 24576:
    print("\nRESULT: Full prompt exceeds configured context.")
    print("Truncation or rejection would be necessary.")
elif len(tokens) > 12290:
    print("\nRESULT: Full prompt exceeds Ollama's reported count.")
    print("This is evidence of possible truncation.")
else:
    print("\nRESULT: Full prompt appears within reported count.")
