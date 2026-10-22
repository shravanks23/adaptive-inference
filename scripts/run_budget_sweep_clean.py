
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------- Configuration --------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "arithmetic_v1.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "budget_sweep_clean_v1.json"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

BUDGETS = [16, 20, 32]
WARMUP_RUNS = 2
MEASURED_RUNS_PER_QUESTION = 3
SEED = 42


# -------------------- Helper Functions --------------------

def synchronize():
    """Wait for pending MPS operations before measuring elapsed time."""
    if DEVICE == "mps":
        torch.mps.synchronize()


def load_dataset():
    """Load the fixed arithmetic evaluation dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("examples", data.get("data", []))

    if not data:
        raise ValueError(
            f"No examples found in dataset: {DATASET_PATH}"
        )

    required_fields = {"id", "question", "answer"}

    for index, item in enumerate(data):
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Dataset item {index} is missing fields: {missing}"
            )

    return data


def extract_answer(text):
    """Extract an unambiguous final integer from the model response."""
    cleaned = text.replace(",", "").strip()

    final_match = re.search(
        r"Final answer\s*:\s*(-?\d+)",
        cleaned,
        re.IGNORECASE,
    )
    if final_match:
        return int(final_match.group(1))

    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)

    result_match = re.search(r"=\s*(-?\d+)\s*$", cleaned)
    if result_match:
        return int(result_match.group(1))

    return None


def prepare_inputs(tokenizer, question):
    """Create a non-thinking Qwen3 chat prompt."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Solve this arithmetic problem: {question}\n"
                "Return exactly one integer: the answer. "
                "Do not show working or any other text."
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )

    return inputs.to(DEVICE)


def generate_once(model, tokenizer, inputs, budget):
    """Generate one response and return its timing, token count, and text."""
    input_tokens = inputs["input_ids"].shape[1]

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
        )

    synchronize()
    elapsed = time.perf_counter() - start

    # Actual generated sequence length, including generated special tokens.
    generated_tokens = output.shape[1] - input_tokens

    response = tokenizer.decode(
        output[0, input_tokens:],
        skip_special_tokens=True,
    ).strip()

    return response, generated_tokens, elapsed


# -------------------- Main Experiment --------------------

def main():
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Loading dataset: {DATASET_PATH}")

    dataset = load_dataset()
    print(f"Dataset examples: {len(dataset)}")

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    model.eval()

    # Prepare prompts once so tokenization is outside timed generation.
    prepared = [
        {
            "item": item,
            "inputs": prepare_inputs(tokenizer, item["question"]),
        }
        for item in dataset
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for budget in BUDGETS:
        print(f"\nTesting max_new_tokens={budget}...")

        # Warm up separately for each budget.
        for _ in range(WARMUP_RUNS):
            generate_once(model, tokenizer, prepared[0]["inputs"], budget)

        records = []
        latencies = []
        total_generated_tokens = 0
        valid_format_count = 0
        correct_count = 0

        for entry in prepared:
            item = entry["item"]
            inputs = entry["inputs"]

            responses = []
            token_counts = []
            question_latencies = []

            for _ in range(MEASURED_RUNS_PER_QUESTION):
                response, generated_tokens, elapsed = generate_once(
                    model,
                    tokenizer,
                    inputs,
                    budget,
                )

                responses.append(response)
                token_counts.append(generated_tokens)
                question_latencies.append(elapsed)

                total_generated_tokens += generated_tokens
                latencies.append(elapsed)

            # Greedy decoding should normally produce identical responses.
            raw_output = responses[-1]
            prediction = extract_answer(raw_output)
            expected = int(item["answer"])

            valid_format = prediction is not None
            is_correct = valid_format and prediction == expected

            valid_format_count += int(valid_format)
            correct_count += int(is_correct)

            records.append(
                {
                    "id": item["id"],
                    "expected": expected,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "generated_tokens_per_run": token_counts,
                    "mean_generated_tokens": (
                        sum(token_counts) / len(token_counts)
                    ),
                    "latency_seconds_per_run": question_latencies,
                    "valid_integer_format": valid_format,
                    "correct": is_correct,
                }
            )

        total_time = sum(latencies)
        total_generations = len(latencies)

        summary = {
            "max_new_tokens": budget,
            "examples": len(dataset),
            "warmup_runs": WARMUP_RUNS,
            "measured_runs_per_question": MEASURED_RUNS_PER_QUESTION,
            "total_generations": total_generations,
            "correct_answers": correct_count,
            "accuracy": correct_count / len(dataset),
            "valid_integer_responses": valid_format_count,
            "valid_format_rate": valid_format_count / len(dataset),
            "total_generated_tokens": total_generated_tokens,
            "mean_tokens_per_generation": (
                total_generated_tokens / total_generations
            ),
            "mean_latency_seconds": sum(latencies) / total_generations,
            "median_latency_seconds": sorted(latencies)[
                len(latencies) // 2
            ],
            "aggregate_tokens_per_second": (
                total_generated_tokens / total_time
                if total_time > 0
                else 0
            ),
            "records": records,
        }

        summaries.append(summary)

        print(
            f"Correct answers: {correct_count}/{len(dataset)}"
        )
        print(
            f"Valid integer responses: "
            f"{valid_format_count}/{len(dataset)}"
        )
        print(
            f"Total generated tokens: {total_generated_tokens}"
        )
        print(
            f"Mean tokens/generation: "
            f"{summary['mean_tokens_per_generation']:.2f}"
        )
        print(
            f"Mean latency: "
            f"{summary['mean_latency_seconds']:.3f} s"
        )
        print(
            f"Aggregate throughput: "
            f"{summary['aggregate_tokens_per_second']:.2f} tokens/s"
        )

    results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dataset": DATASET_PATH.name,
        "seed": SEED,
        "summaries": summaries,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()