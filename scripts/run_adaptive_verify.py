import json
import re
import time
from fractions import Fraction
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
REPEATS = 3

OUTPUT_PATH = RESULTS_DIR / "adaptive_verify_v1.json"


# -------------------- Independent Arithmetic Verifier --------------------

def verify_answer(question, prediction):
    """
    Verify an integer prediction by independently evaluating
    the two-operand arithmetic expression in the question.

    This verifier is specific to our arithmetic pilot.
    It does not use the dataset's expected answer.
    """
    matches = re.findall(
        r"(\d+)\s*([+\-*/])\s*(\d+)",
        question,
    )

    if not matches or prediction is None:
        return {
            "verified": False,
            "computed_answer": None,
        }

    left_text, operator, right_text = matches[-1]
    left = int(left_text)
    right = int(right_text)

    if operator == "+":
        answer = Fraction(left + right)
    elif operator == "-":
        answer = Fraction(left - right)
    elif operator == "*":
        answer = Fraction(left * right)
    elif operator == "/":
        if right == 0:
            return {
                "verified": False,
                "computed_answer": None,
            }
        answer = Fraction(left, right)
    else:
        return {
            "verified": False,
            "computed_answer": None,
        }

    if answer.denominator != 1:
        return {
            "verified": False,
            "computed_answer": None,
        }

    computed_answer = answer.numerator

    return {
        "verified": prediction == computed_answer,
        "computed_answer": computed_answer,
    }


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
        "prediction": extract_answer(response),
        "tokens": len(generated_ids),
        "generation_seconds": elapsed,
        "emitted_eos": emitted_eos,
        "hit_cap": hit_cap,
    }

    del output
    return result


# -------------------- Policy Execution --------------------

def run_policy(model, tokenizer, item, inputs, policy):
    question = item["question"]
    verifier_seconds = 0.0
    attempts = []

    if policy == "fixed_13":
        budgets = [START_BUDGET]
    elif policy == "fixed_14":
        budgets = [ESCALATED_BUDGET]
    elif policy == "adaptive_verify_then_escalate":
        budgets = [START_BUDGET]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    for budget in budgets:
        attempt = generate(model, tokenizer, inputs, budget)

        # Only the adaptive policy uses the verifier to decide
        # whether additional generation is necessary.
        if policy == "adaptive_verify_then_escalate":
            verify_start = time.perf_counter()
            verification = verify_answer(
                question, attempt["prediction"]
            )
            verify_elapsed = time.perf_counter() - verify_start

            attempt["verification"] = verification
            attempt["verifier_seconds"] = verify_elapsed
            verifier_seconds += verify_elapsed

        attempts.append(attempt)

    escalated = False

    if (
        policy == "adaptive_verify_then_escalate"
        and not attempts[0]["verification"]["verified"]
    ):
        escalated = True

        second = generate(
            model, tokenizer, inputs, ESCALATED_BUDGET
        )

        # Verify the retry too, for diagnostic purposes.
        verify_start = time.perf_counter()
        verification = verify_answer(
            question, second["prediction"]
        )
        verify_elapsed = time.perf_counter() - verify_start

        second["verification"] = verification
        second["verifier_seconds"] = verify_elapsed
        verifier_seconds += verify_elapsed

        attempts.append(second)

    final_attempt = attempts[-1]
    prediction = final_attempt["prediction"]
    expected = int(item["answer"])

    generation_seconds = sum(
        attempt["generation_seconds"] for attempt in attempts
    )

    return {
        "prediction": prediction,
        "correct": prediction == expected,
        "tokens_used": sum(
            attempt["tokens"] for attempt in attempts
        ),
        "generation_seconds": generation_seconds,
        "verifier_seconds": verifier_seconds,
        "total_seconds": generation_seconds + verifier_seconds,
        "escalated": escalated,
        "attempts": attempts,
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
        "adaptive_verify_then_escalate",
    ]

    results = {}

    for policy in policies:
        print(f"\nRunning {policy}...")

        for _ in range(WARMUP_RUNS):
            run_policy(
                model,
                tokenizer,
                prepared[0][0],
                prepared[0][1],
                policy,
            )

        records = []
        all_latencies = []
        total_tokens = 0
        total_correct = 0
        total_escalations = 0
        total_verifier_seconds = 0.0
        total_generation_seconds = 0.0

        for item, inputs in prepared:
            runs = []

            for _ in range(REPEATS):
                result = run_policy(
                    model, tokenizer, item, inputs, policy
                )

                runs.append(result)
                total_tokens += result["tokens_used"]
                total_correct += int(result["correct"])
                total_escalations += int(result["escalated"])

                total_generation_seconds += (
                    result["generation_seconds"]
                )
                total_verifier_seconds += (
                    result["verifier_seconds"]
                )
                all_latencies.append(result["total_seconds"])

            records.append({
                "id": item["id"],
                "expected": int(item["answer"]),
                "runs": runs,
            })

        total_runs = len(dataset) * REPEATS
        sorted_latencies = sorted(all_latencies)

        summary = {
            "examples": len(dataset),
            "measured_runs_per_question": REPEATS,
            "total_runs": total_runs,
            "correct_runs": total_correct,
            "accuracy": total_correct / total_runs,
            "total_generated_tokens": total_tokens,
            "mean_tokens_per_run": total_tokens / total_runs,
            "mean_total_latency_seconds": (
                sum(all_latencies) / total_runs
            ),
            "median_total_latency_seconds": sorted_latencies[
                len(sorted_latencies) // 2
            ],
            "total_generation_seconds": total_generation_seconds,
            "total_verifier_seconds": total_verifier_seconds,
            "mean_verifier_seconds_per_run": (
                total_verifier_seconds / total_runs
            ),
            "escalations": total_escalations,
            "records": records,
        }

        results[policy] = summary

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
            f"Mean total latency: "
            f"{summary['mean_total_latency_seconds']:.4f}s"
        )
        print(
            f"Total verifier time: "
            f"{total_verifier_seconds:.6f}s"
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
