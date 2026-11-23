# Needle in a Haystack: Evaluating Long-Context Retrieval with Qwen3

A reproducible, local **LLM evaluation project** that investigates whether a language model can retrieve a known fact from different positions in increasingly long prompts—and how much time those requests take.

This project runs **Qwen3:14B** through **Ollama** on an Apple Silicon Mac, measures input size with the Qwen tokenizer, checks Ollama's reported prompt-token count, and records retrieval accuracy and latency.

> **Scope:** This is an independent educational benchmark inspired by the *needle-in-a-haystack* evaluation methodology. It is not a claim of a new benchmark or a comprehensive assessment of long-context reasoning.

## Research questions

1. Does a model retrieve a clearly labeled fact when it appears near the **beginning**, **middle**, or **end** of a long document?
2. Does retrieval success change as **input context length** increases?
3. How does **end-to-end latency** vary with longer prompts?
4. How can **input truncation and evaluation-scoring errors** distort apparent model performance?

## Evaluation pipeline

```text
Generate varied synthetic research notes
                |
Insert unique ATLAS-#### verification code
                |
Count full input using Qwen tokenizer
                |
Check input fits allocated context + output budget
                |
Call local Ollama API (Qwen3:14B)
                |
Compare Ollama prompt tokens with tokenizer count
                |
Score retrieval + exact output format; record latency
                |
Save CSV results
```

The hidden fact looks like:

```text
The secret verification code for Project Atlas is ATLAS-6378.
```

The model is instructed to respond with only the code. In each test, the expected code is known in advance. A **retrieval pass** requires the response to contain exactly one `ATLAS-####` code and for that code to match the expected value. A separate **format pass** requires the entire response to equal the code exactly. Invalid or truncated input is **not** scored as a model failure.

## Hardware and software

The reported results below came from:

| Component | Configuration |
|---|---|
| Computer | MacBook Pro, Apple M1 Max, 64 GB unified memory |
| Inference engine | Ollama 0.40.11, llama.cpp runner |
| Model | `qwen3:14b` (14.8B parameters, Q4_K_M quantization) |
| Model-reported maximum context | 40,960 tokens |
| Token counter | Hugging Face `Qwen/Qwen3-14B` tokenizer |
| Implementation | Python, `requests`, `transformers`, `tokenizers` |
| Evaluation conditions | `temperature=0`, `think=False` |

The configured **runtime** context was 24,576 tokens for the 4K–10K tests and **40,960 tokens** for the 24K pilot. The model's reported maximum and the runtime allocation are different things.

## Results

### Token-validated retrieval

| Target input length | Needle at 0% | Needle at 50% | Needle at 100% | Total |
|---|---:|---:|---:|---:|
| ~4K tokens | 5/5 | 5/5 | 5/5 | **15/15** |
| ~8K tokens | 5/5 | 5/5 | 5/5 | **15/15** |
| ~10K tokens | 5/5 | 5/5 | 5/5 | **15/15** |
| ~24K tokens | 1/1 | 1/1 | 1/1 | **3/3** |
| **Total** | **16/16** | **16/16** | **16/16** | **48/48** |

These are **observed successes in a synthetic single-fact retrieval task**, not evidence of perfect long-context reasoning. The 24K condition has only **one trial per position**.

### Input token integrity

| Target | Hugging Face token count | Ollama reported prompt tokens | Difference |
|---|---:|---:|---:|
| 4K example | 3,932 | 3,948 | +16 |
| 8K example | 7,975 | 7,991 | +16 |
| 10K example | 9,928 | 9,944 | +16 |
| 24K pilot | 23,946 | 23,962 | +16 |

The small, consistent difference is compatible with inference-template or token-accounting overhead. These measurements provide evidence that the validated prompts were processed substantially as intended, but do not independently prove every internal token-handling detail.

### 24K latency pilot

| Needle position | End-to-end response time | Result |
|---|---:|---|
| 0% — beginning | 247.00 s | Pass |
| 50% — middle | 174.46 s | Pass |
| 100% — end | 187.59 s | Pass |
| **Mean** | **203.02 s (~3 min 23 s)** | **3/3** |

These are individual wall-clock observations, not controlled latency benchmarks; cache state, prompt-processing time, model loading, and execution order may influence them. They do **not** establish that needle position directly causes changes in latency.

## An important debugging finding: oversized prompts

An earlier version of the test generated a prompt containing **52,047 tokens** (independently counted), while Ollama's configured context was only **24,576 tokens**. Ollama reported **12,290 processed tokens**; codes at the beginning and middle were missed, while an end-position code was retrieved.

That experiment was **invalid as a test of full-document retrieval**. Its failures cannot be attributed to the model's attention or reasoning ability because the complete input did not fit the configured context. The benchmark was subsequently redesigned to enforce a token budget and flag substantial discrepancies between independent and runtime token counts.

A second issue was fixed in scoring: a response such as `The code is **ATLAS-1234**` contains the correct fact but fails strict whole-response string equality. Retrieval correctness and format compliance are now tracked separately.

**Lesson:** Validate the evaluation harness before drawing conclusions about the model.

## Getting started

### Prerequisites

- Python 3.10+ (a newer Python version may also work)
- [Ollama](https://ollama.com/) installed and running
- Enough available memory for Qwen3:14B and the chosen context allocation

Download the model:

```bash
ollama pull qwen3:14b
ollama list
```

Check that the Ollama API is reachable:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install requests transformers tokenizers
```

Only the **tokenizer** is loaded via Transformers; the model itself runs through Ollama. The warning that PyTorch is not installed is harmless for this setup.

### Run the token-budgeted benchmark

The main benchmark is `benchmark_v3.py`. Check the constants near the top of the script (`TARGETS`, `DEPTHS`, `TRIALS`, `NUM_CTX`, `MAX_ALLOWED_INPUT_TOKENS`, and `OUTPUT`) before execution.

Run a **dry run first**—this creates and validates test prompts without model inference:

```bash
python benchmark_v3.py --dry-run
```

Then run the benchmark:

```bash
python benchmark_v3.py
```

The benchmark writes incremental CSV results to the filename configured by `OUTPUT` (normally `results_v3.csv`). It checks that the prompt is within the configured token budget before sending it to Ollama. If Ollama reports a major token-count mismatch, it marks the input invalid and stops to avoid producing misleading scores.

For the **45-trial reliability experiment**, use:

```python
TARGETS = [4000, 8000, 10000]
DEPTHS = [0, 50, 100]
TRIALS = 5
NUM_CTX = 24576
MAX_ALLOWED_INPUT_TOKENS = 11000
OUTPUT = Path("results_v3.csv")
```

For the **24K pilot**, use the separate `benchmark_24k.py` if present, or configure the same benchmark as follows:

```python
TARGETS = [24000]
DEPTHS = [0, 50, 100]
TRIALS = 1
NUM_CTX = 40960
MAX_ALLOWED_INPUT_TOKENS = 26000
OUTPUT = Path("results_24k.csv")
```

The values above are **configuration snippets**, not shell commands. They belong in the script's Python configuration section. Increasing context length can increase memory consumption and latency substantially.

### Results files

The following files are useful to publish **if they exist in your local project and have been reviewed**:

- `results_v3_45_trials.csv`: 45-run validated reliability experiment.
- `results_24k.csv`: three-run 24K pilot.
- `check_tokens.py`: independent tokenizer diagnostic, if retained.
- `plot.py` or `plot_results.py`: plotting/analysis script, if retained.
- `benchmark.py` and `benchmark_v2.py`: historical experiments, useful for explaining methodology failures; not the recommended production benchmark.

Do not commit `.venv/`, `.env`, API keys, downloaded model weights, or confidential documents.

## Experimental limitations

- **Synthetic benchmark:** Varied generated research notes are not equivalent to real legal, medical, or engineering documents.
- **Easy retrieval target:** Extracting a single explicitly named code is much easier than resolving contradictory evidence or multi-hop reasoning.
- **Limited sampling:** Five trials per condition at 4K–10K; only one trial per position at 24K.
- **Single model and hardware:** No cross-model or cross-device comparison.
- **Partial reproducibility:** Randomized secret codes are not fixed across runs; the benchmark randomizes execution order with a fixed seed, but exact runs may differ.
- **Latency confounders:** KV caching, warming, background load, and prompt length may change timing.
- **Tested range:** These results do not establish behavior at 30K, 38K, or the model's full advertised context capacity.

## Potential next experiments

1. **Distractor resistance:** Include multiple plausible codes, timestamps, and stale records; evaluate selection of the correct one.
2. **Multi-hop retrieval and reasoning:** Require combining information from distant passages.
3. **Longer validated contexts:** Test ~30K and ~38K while ensuring prompt+output fit the allocated context.
4. **Full context vs. RAG:** Compare direct long-prompt inference with retrieval of relevant passages, including retrieval failure rates.
5. **Cross-model comparison:** Run identical, seeded test sets on additional local models with matched settings.

## Prior work and acknowledgments

The experiment was motivated by [LeonEricsson/llmcontext](https://github.com/LeonEricsson/llmcontext) and the broader needle-in-a-haystack evaluation approach. This repository provides an independent Ollama-based implementation and reports experiments run locally. Please see the upstream project for its implementation and attribution.

## License

No license is asserted by this README. If you intend others to reuse or distribute your code, add an appropriate `LICENSE` file after checking the provenance and licenses of any included third-party code.
