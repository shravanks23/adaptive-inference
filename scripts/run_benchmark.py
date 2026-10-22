
import json
import re
import statistics
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "arithmetic_v1.json"
RESULTS_PATH = ROOT / "results" / "benchmark_v1.json"

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 2
MEASURED_RUNS = 5


def synchronize():
    """Wait for pending MPS operations before measuring time."""
    if DEVICE == "mps":
        torch.mps.synchronize()


def load_dataset():
    """Load the fixed arithmetic evaluation dataset."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    required_fields = {"id", "question", "answer"}
    for item in dataset:
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Dataset item {item.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return dataset


def extract_answer(text):
    """Extract the last integer from the model's response."""
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def prepare_inputs(tokenizer, question):
    """Apply the model's chat template and prepare device inputs."""
    messages = [
        {
            "role": "user",
            "content": f"{question} Reply with only the final number.",
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)


def generate_once(model, inputs, tokenizer):
    """Generate one answer and return output text, token count, and latency."""
    input_tokens = inputs["input_ids"].shape[1]

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = output.shape[1] - input_tokens
    answer_text = tokenizer.decode(
        output[0, input_tokens:],
        skip_special_tokens=True,
    )

    return answer_text, generated_tokens, elapsed


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    dataset = load_dataset()
    process = psutil.Process()

    # Warm-up generations are excluded from measured results.
    warmup_inputs = prepare_inputs(
        tokenizer,
        "What is 17 + 28?",
    )

    print(f"Running {WARMUP_RUNS} warm-up runs...")
    for _ in range(WARMUP_RUNS):
        generate_once(model, warmup_inputs, tokenizer)

    records = []
    run_latencies = []
    total_correct = 0
    total_generated_tokens = 0
    total_input_tokens = 0
    total_generations = 0

    for item in dataset:
        inputs = prepare_inputs(tokenizer, item["question"])
        input_tokens = inputs["input_ids"].shape[1]

        run_outputs = []
        run_token_counts = []
        latencies = []

        for _ in range(MEASURED_RUNS):
            output_text, generated_tokens, elapsed = generate_once(
                model,
                inputs,
                tokenizer,
            )

            run_outputs.append(output_text)
            run_token_counts.append(generated_tokens)
            latencies.append(elapsed)

        # Score the final measured response and preserve all run measurements.
        final_output = run_outputs[-1]
        predicted = extract_answer(final_output)
        correct = predicted == str(item["answer"])

        total_correct += int(correct)
        total_generated_tokens += sum(run_token_counts)
        total_input_tokens += input_tokens * MEASURED_RUNS
        total_generations += MEASURED_RUNS
        run_latencies.extend(latencies)

        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": str(item["answer"]),
                "prediction": predicted,
                "raw_output": final_output,
                "correct": correct,
                "input_tokens_per_run": input_tokens,
                "generated_tokens_per_run": run_token_counts,
                "latency_seconds_runs": latencies,
                "latency_seconds_mean": statistics.mean(latencies),
            }
        )

        print(
            f"{item['id']}: prediction={predicted}, "
            f"expected={item['answer']}, correct={correct}"
        )

    total_questions = len(dataset)
    total_time = sum(run_latencies)

    results = {
        "experiment": "qwen3_arithmetic_baseline_v1",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs_per_question": MEASURED_RUNS,
        "total_measured_generations": total_generations,
        "examples": total_questions,
        "accuracy": total_correct / total_questions,
        "correct_answers": total_correct,
        "mean_generation_latency_seconds": statistics.mean(run_latencies),
        "median_generation_latency_seconds": statistics.median(run_latencies),
        "total_measured_generation_seconds": total_time,
        "aggregate_generated_tokens_per_second": (
            total_generated_tokens / total_time if total_time > 0 else None
        ),
        "total_input_tokens_all_runs": total_input_tokens,
        "total_generated_tokens_all_runs": total_generated_tokens,
        "process_rss_bytes_after_run": process.memory_info().rss,
        "measurement_note": (
            "Pilot benchmark. Greedy decoding; accuracy is scored on the "
            "final measured response for each question. Latency excludes "
            "model loading and tokenization. Throughput uses all generated "
            "tokens divided by all measured generation time. RSS is process "
            "resident memory and may not capture all MPS memory usage."
        ),
        "records": records,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n--- Benchmark summary ---")
    print(f"Accuracy: {total_correct}/{total_questions}")
    print(
        f"Mean generation latency: "
        f"{results['mean_generation_latency_seconds']:.3f} s"
    )
    print(
        f"Median generation latency: "
        f"{results['median_generation_latency_seconds']:.3f} s"
    )
    print(
        f"Aggregate throughput: "
        f"{results['aggregate_generated_tokens_per_second']:.2f} tokens/s"
    )
    print(f"Measured generations: {total_generations}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()