import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "arithmetic_v2.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "budget_sweep_v2.json"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

BUDGETS = [16, 24, 32, 48]
WARMUP_RUNS = 2
MEASURED_RUNS_PER_QUESTION = 3


def synchronize():
    if DEVICE == "mps":
        torch.mps.synchronize()


def extract_answer(text):
    cleaned = text.replace(",", "").strip()

    match = re.search(
        r"Final answer\s*:\s*(-?\d+)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)

    match = re.search(r"=\s*(-?\d+)\s*$", cleaned)
    if match:
        return int(match.group(1))

    return None


def prepare_inputs(tokenizer, question):
    messages = [{
        "role": "user",
        "content": (
            f"Solve this arithmetic problem: {question}\n"
            "Return exactly one integer: the answer. "
            "Do not show working or any other text."
        ),
    }]

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

    generated_tokens = output.shape[1] - input_tokens
    response = tokenizer.decode(
        output[0, input_tokens:],
        skip_special_tokens=True,
    ).strip()

    return response, generated_tokens, elapsed


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    required = {"id", "question", "expected"}
    for item in dataset:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item.get('id', '?')} missing: {missing}")

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH.name} ({len(dataset)} questions)")
    print(f"Budgets: {BUDGETS}")
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    prepared = [
        {
            "item": item,
            "inputs": prepare_inputs(tokenizer, item["question"]),
        }
        for item in dataset
    ]

    summaries = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for budget in BUDGETS:
        print(f"\nTesting max_new_tokens={budget}...")

        for _ in range(WARMUP_RUNS):
            generate_once(
                model, tokenizer, prepared[0]["inputs"], budget
            )

        records = []
        latencies = []
        total_tokens = 0
        correct_count = 0
        valid_count = 0
        eos_count = 0
        cap_count = 0

        for entry in prepared:
            item = entry["item"]
            inputs = entry["inputs"]
            runs = []

            for _ in range(MEASURED_RUNS_PER_QUESTION):
                response, tokens, elapsed = generate_once(
                    model, tokenizer, inputs, budget
                )
                prediction = extract_answer(response)
                correct = prediction == int(item["expected"])

                runs.append({
                    "response": response,
                    "prediction": prediction,
                    "tokens": tokens,
                    "latency_seconds": elapsed,
                    "correct": correct,
                })

                total_tokens += tokens
                latencies.append(elapsed)

                if prediction is not None:
                    valid_count += 1
                if correct:
                    correct_count += 1

            # Inspect the final generated sequence to determine whether
            # generation ended with EOS or reached the configured cap.
            # Repeated greedy outputs are retained for reproducibility.
            final_response = runs[-1]["response"]
            final_prediction = runs[-1]["prediction"]
            records.append({
                "id": item["id"],
                "question": item["question"],
                "expected": int(item["expected"]),
                "prediction": final_prediction,
                "raw_output": final_response,
                "correct": final_prediction == int(item["expected"]),
                "runs": runs,
            })

        generations = len(latencies)
        summary = {
            "max_new_tokens": budget,
            "examples": len(dataset),
            "measured_runs_per_question": MEASURED_RUNS_PER_QUESTION,
            "total_generations": generations,
            "correct_runs": correct_count,
            "accuracy": correct_count / generations,
            "valid_integer_runs": valid_count,
            "valid_integer_rate": valid_count / generations,
            "total_generated_tokens": total_tokens,
            "mean_tokens_per_generation": total_tokens / generations,
            "mean_latency_seconds": sum(latencies) / generations,
            "records": records,
        }
        summaries.append(summary)

        print(f"Correct runs: {correct_count}/{generations}")
        print(f"Valid integer runs: {valid_count}/{generations}")
        print(f"Total generated tokens: {total_tokens}")
        print(f"Mean tokens/generation: {summary['mean_tokens_per_generation']:.2f}")
        print(f"Mean latency: {summary['mean_latency_seconds']:.3f} s")

    results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dataset": DATASET_PATH.name,
        "warmup_runs": WARMUP_RUNS,
        "summaries": summaries,
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
