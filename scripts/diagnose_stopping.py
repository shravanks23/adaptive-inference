import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

question = "1000 - 476"
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE
).to(DEVICE)
model.eval()

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to(DEVICE)

eos_ids = model.generation_config.eos_token_id
if eos_ids is None:
    eos_ids = []
elif isinstance(eos_ids, int):
    eos_ids = [eos_ids]

print(f"Device: {DEVICE}")
print(f"EOS token IDs: {eos_ids}")

for budget in (13, 14):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            return_dict_in_generate=True,
        )

    new_ids = output.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    emitted_eos = any(token_id in eos_ids for token_id in new_ids)
    hit_cap = len(new_ids) >= budget

    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    print(f"\n--- Budget: {budget} ---")
    print(f"Generated token count: {len(new_ids)}")
    print(f"Reached token cap: {hit_cap}")
    print(f"EOS token emitted: {emitted_eos}")
    print(f"Token IDs: {new_ids}")
    print(f"Raw output: {decoded!r}")

    del output
