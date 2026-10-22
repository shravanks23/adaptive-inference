
import json
import time
from pathlib import Path

import random
import statistics


SEED = 42
NUM_EXAMPLES = 100
NUM_REPEATS = 5


def generate_dataset():
    rng = random.Random(SEED)
    dataset = []

    for _ in range(NUM_EXAMPLES):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        dataset.append({
            "a": a,
            "b": b,
            "answer": a + b,
        })

    return dataset


def run_fixed_baseline(dataset):
    predictions = []
    start = time.perf_counter()

    for item in dataset:
        predictions.append(item["a"] + item["b"])

    elapsed = time.perf_counter() - start
    correct = sum(
        prediction == item["answer"]
        for prediction, item in zip(predictions, dataset)
    )

    return {
        "accuracy": correct / len(dataset),
        "latency_seconds": elapsed,
        "examples": len(dataset),
        "method": "fixed_baseline",
    }


def main():
    dataset = generate_dataset()
    measurements = []

    for _ in range(NUM_REPEATS):
        measurements.append(run_fixed_baseline(dataset))

    result = {
        "experiment": "synthetic_addition_baseline",
        "seed": SEED,
        "repeats": NUM_REPEATS,
        "accuracy_mean": statistics.mean(
            item["accuracy"] for item in measurements
        ),
        "latency_seconds_mean": statistics.mean(
            item["latency_seconds"] for item in measurements
        ),
        "examples": NUM_EXAMPLES,
        "note": "Synthetic pipeline test; not an LLM benchmark.",
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "baseline.json"
    output_file.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()