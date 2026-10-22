
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )
    model = model.to(DEVICE)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": (
                "Explain why measuring AI inference latency matters. "
                "Give a concise answer in 3 bullet points."
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
    ).to(DEVICE)

    if DEVICE == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    generated_tokens = (
        output.shape[1] - inputs["input_ids"].shape[1]
    )

    print("\n--- Model answer ---")
    print(answer)
    print("\n--- Measurements ---")
    print(f"Device: {DEVICE}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print(f"Tokens per second: {generated_tokens / elapsed:.2f}")


if __name__ == "__main__":
    main()