
import json
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "arithmetic_v1.json"
RESULTS_PATH = ROOT / "results" / "budget_sweep_v1.json"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BUDGETS = [16, 20, 32]
WARMUP_RUNS = 2
MEASURED_RUNS = 3


def synchronize():
    if DEVICE == "mps":
        torch.mps.synchronize()




def extract_answer(text):
    """Extract the answer from a labelled response or arithmetic working."""
    cleaned = text.replace(",", "").strip()

    # Prefer an explicitly labelled final answer.
    final_match = re.search(
        r"Final answer\s*:\s*(-?\d+)",
        cleaned,
        re.IGNORECASE,
    )
    if final_match:
        return int(final_match.group(1))

    # If the response ends with '=', it is clearly unfinished.
    if cleaned.endswith("="):
        return None

    # Extract the last integer from the response.
    matches = re.findall(r"-?\d+", cleaned)
    return int(matches[-1]) if matches else None

def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Dataset is empty.")

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    ).to(DEVICE)
    model.eval()

    prepared = []

    for item in dataset:
        messages = [{
            "role": "user",
            "content": (
                f"{item['question']} "
                "Reply with only the final number."
            ),
        }]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(DEVICE)

        prepared.append((item, inputs))

    print(f"Running {WARMUP_RUNS} warm-up generations...")

    for _ in range(WARMUP_RUNS):
        with torch.inference_mode():
            model.generate(
                **prepared[0][1],
                max_new_tokens=32,
                do_sample=False,
            )
        synchronize()

    summaries = []

    for budget in BUDGETS:
        print(f"\nTesting max_new_tokens={budget}...")

        correct = 0
        latencies = []
        generated_token_total = 0
        measured_count = 0
        records = []

        for item, inputs in prepared:
            input_tokens = inputs["input_ids"].shape[1]
            outputs = []

            for _ in range(MEASURED_RUNS):
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
                latencies.append(elapsed)

                token_count = output.shape[1] - input_tokens
                generated_token_total += token_count
                measured_count += 1

                text = tokenizer.decode(
                    output[0, input_tokens:],
                    skip_special_tokens=True,
                ).strip()

                outputs.append(text)

            # Greedy decoding should be deterministic under identical settings.
            prediction_text = outputs[-1]
            prediction = extract_answer(prediction_text)
            expected = int(item["answer"])
            is_correct = prediction == expected
            correct += int(is_correct)

            records.append({
                "id": item["id"],
                "expected": expected,
                "prediction": prediction,
                "raw_output": prediction_text,
                "correct": is_correct,
            })

        elapsed_total = sum(latencies)

        summary = {
            "max_new_tokens": budget,
            "examples": len(dataset),
            "measured_runs_per_question": MEASURED_RUNS,
            "measured_generations": measured_count,
            "accuracy": correct / len(dataset),
            "correct_answers": correct,
            "mean_latency_seconds": statistics.mean(latencies),
            "median_latency_seconds": statistics.median(latencies),
            "total_generation_seconds": elapsed_total,
            "total_generated_tokens": generated_token_total,
            "aggregate_tokens_per_second": (
                generated_token_total / elapsed_total
                if elapsed_total > 0 else None
            ),
            "records": records,
        }

        summaries.append(summary)

        print(f"Accuracy: {correct}/{len(dataset)}")
        print(
            f"Mean latency: "
            f"{summary['mean_latency_seconds']:.3f} s"
        )
        print(
            "Aggregate throughput: "
            f"{summary['aggregate_tokens_per_second']:.2f} tokens/s"
        )

    results = {
        "experiment": "qwen3_token_budget_sweep_v2",
        "model": MODEL_NAME,
        "device": DEVICE,
        "decoding": "greedy",
        "warmup_runs": WARMUP_RUNS,
        "summaries": summaries,
        "note": (
            "Small arithmetic pilot. Accuracy is scored on the final "
            "measured response per question using labelled-answer-first "
            "extraction. Token limits are maximums, not guaranteed token "
            "counts. Loading and tokenization are excluded from latency."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()