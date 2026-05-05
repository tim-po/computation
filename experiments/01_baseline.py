"""
Baseline: скачать Qwen3.5-0.8B, запустить на MPS, измерить tok/s.
Только текстовая часть модели (игнорируем vision encoder).

Запуск: python experiments/01_baseline.py
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number recursively.",
    "Explain the difference between TCP and UDP in simple terms.",
    "What is the capital of France?",
]


def tok_per_sec(n_tokens: int, elapsed: float) -> float:
    return n_tokens / elapsed


def run_baseline():
    print(f"Device: {DEVICE}")
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model (bfloat16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Параметры
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    text_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"Total params: {total_params:.1f}M  |  Text model: {text_params:.1f}M")

    # Тест warmup
    print("\n--- Warmup ---")
    inp = tokenizer("Hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=5, do_sample=False)
    print("Warmup done.")

    # Бенчмарк
    results = []
    print("\n--- Benchmark (greedy, max_new_tokens=100) ---")
    for prompt in PROMPTS:
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        n_in = inp["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        elapsed = time.time() - t0

        n_out = out.shape[1] - n_in
        tps = tok_per_sec(n_out, elapsed)
        results.append(tps)

        response = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"Generated ({n_out} tokens in {elapsed:.2f}s = {tps:.1f} tok/s):")
        print(f"  {response[:120]}")

    print(f"\n=== RESULTS ===")
    print(f"tok/s: min={min(results):.1f}  max={max(results):.1f}  avg={sum(results)/len(results):.1f}")
    print(f"Device: {DEVICE} | Model: {MODEL_ID}")


if __name__ == "__main__":
    run_baseline()
