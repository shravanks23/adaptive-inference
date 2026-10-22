
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_budget_sweep_clean import (
    DEVICE,
    DTYPE,
    MODEL_NAME,
    RESULTS_DIR,
    load_dataset,
    prepare_inputs,
    extract_answer,
)


# -------------------- Configuration --------------------

START_BUDGET = 13
ESCALATED_BUDGET = 14
WARMUP_RUNS = 2
MEASURED_RUNS_PER_QUESTION = 3

OUTPUT_PATH = RESULTS_DIR / "adaptive_pilot_v1.json"


# -------------------- Generation --------------------

def synchronize():
    if DEVICE == "mps":
        torch.mps.synchronize()


def generate(model, tokenizer, inputs, budget):
    input_length = inputs["input_ids"].shape[1]

    synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            return_dict_in_generate=True,
        )

    synchronize()
    elapsed = time.perf_counter() - start_time

    generated_ids = output.sequences[
        0, input_length:
    ].tolist()

    eos_ids = model.generation_config.eos_token_id

    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    emitted_eos = any(
        token_id in eos_ids for token_id in generated_ids
    )

    hit_cap = (
        len(generated_ids) >= budget and not emitted_eos
    )

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    result = {
        "response": response,
        "tokens": len(generated_ids),
        "latency": elapsed,
        "emitted_eos": emitted_eos,
        "hit_cap": hit_cap,
    }

    del output
    return result


# -------------------- Policies --------------------

def run_policy(model, tokenizer, inputs, policy):
    if policy == "fixed_13":
        attempts = [
            generate(model, tokenizer, inputs, START_BUDGET)
        ]

    elif policy == "fixed_14":
        attempts = [
            generate(model, tokenizer, inputs, ESCALATED_BUDGET)
        ]

    elif policy == "adaptive_13_to_14":
        first = generate(
            model, tokenizer, inputs, START_BUDGET
        )
        attempts = [first]

        if first["hit_cap"]:
            second = generate(
                model, tokenizer, inputs, ESCALATED_BUDGET
            )
            attempts.append(second)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    final_response = attempts[-1]["response"]

    return {
        "response": final_response,
        "prediction": extract_answer(final_response),
        "tokens_used": sum(
            attempt["tokens"] for attempt in attempts
        ),
        "latency_seconds": sum(
            attempt["latency"] for attempt in attempts
        ),
        "escalated": len(attempts) > 1,
        "attempts": [
            {
                "response": attempt["response"],
                "tokens": attempt["tokens"],
                "emitted_eos": attempt["emitted_eos"],
                "hit_cap": attempt["hit_cap"],
            }
            for attempt in attempts
        ],
    }


# -------------------- Experiment --------------------

def main():
    torch.manual_seed(42)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    dataset = load_dataset()
    print(f"Dataset examples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    ).to(DEVICE)

    model.eval()

    prepared = [
        (item, prepare_inputs(tokenizer, item["question"]))
        for item in dataset
    ]

    policies = [
        "fixed_13",
        "fixed_14",
        "adaptive_13_to_14",
    ]

    results = {}

    for policy_name in policies:
        print(f"\nRunning {policy_name}...")

        # Warm up the generation path.
        for _ in range(WARMUP_RUNS):
            run_policy(
                model,
                tokenizer,
                prepared[0][1],
                policy_name,
            )

        records = []
        all_latencies = []
        total_tokens = 0
        total_correct = 0
        total_escalations = 0

        for item, inputs in prepared:
            expected = int(item["answer"])
            runs = []

            for _ in range(MEASURED_RUNS_PER_QUESTION):
                result = run_policy(
                    model,
                    tokenizer,
                    inputs,
                    policy_name,
                )

                prediction = result["prediction"]
                result["correct"] = prediction == expected

                runs.append(result)
                all_latencies.append(
                    result["latency_seconds"]
                )
                total_tokens += result["tokens_used"]
                total_correct += int(result["correct"])
                total_escalations += int(result["escalated"])

            records.append({
                "id": item["id"],
                "expected": expected,
                "runs": runs,
            })

        total_runs = len(dataset) * MEASURED_RUNS_PER_QUESTION
        sorted_latencies = sorted(all_latencies)

        summary = {
            "examples": len(dataset),
            "measured_runs_per_question": (
                MEASURED_RUNS_PER_QUESTION
            ),
            "total_runs": total_runs,
            "correct_runs": total_correct,
            "accuracy": total_correct / total_runs,
            "total_generated_tokens": total_tokens,
            "mean_tokens_per_run": total_tokens / total_runs,
            "mean_latency_seconds": (
                sum(all_latencies) / total_runs
            ),
            "median_latency_seconds": sorted_latencies[
                len(sorted_latencies) // 2
            ],
            "escalations": total_escalations,
            "records": records,
        }

        results[policy_name] = summary

        print(
            f"Accuracy: {total_correct}/{total_runs} "
            f"({summary['accuracy']:.1%})"
        )
        print(f"Generated tokens: {total_tokens}")
        print(
            f"Mean tokens/run: "
            f"{summary['mean_tokens_per_run']:.2f}"
        )
        print(
            f"Mean latency: "
            f"{summary['mean_latency_seconds']:.4f}s"
        )
        print(f"Escalations: {total_escalations}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()