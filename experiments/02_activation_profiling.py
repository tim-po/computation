"""
Activation profiling для Qwen3.5-0.8B.

Что измеряем:
  1. Residual magnitude per layer: ||h_out - h_in||_F / ||h_in||_F
     — насколько сильно каждый слой меняет hidden state
     — слои с большим residual важнее и пострадают сильнее от split без синхронизации

  2. Head contribution (только full_attention слои): сколько каждая
     группа attention голов вносит в итоговый output O-proj?
     — если головы вносят равный вклад: split погрешность ≈ (N-1)/N
     — если головы неравны: split может быть лучше или хуже

  3. Накопленная δ через прокси residual:
     — CCA точки: перед слоями с наибольшими residual

Запуск: python experiments/02_activation_profiling.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number recursively.",
    "Explain the difference between TCP and UDP in simple terms.",
    "In a study of 1000 patients, 450 received treatment A, 350 received treatment B. Which is more effective?",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Solve step by step: 2x + 5 = 13. What is x?",
    "The quick brown fox jumps over the lazy dog. " * 10 + "What animal jumped?",
    "Translate to French: Hello, how are you today?",
    "What is the derivative of f(x) = x^3 + 2x^2 - 5?",
]


def frobenius_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm()
    return (diff / (denom + 1e-8)).item()


def collect_layer_residuals(
    model, tokenizer, prompts: list[str], device: str
) -> dict[int, float]:
    """
    Для каждого слоя замеряем: насколько сильно слой меняет hidden state.
    residual_ratio[l] = E[||h_out - h_in||_F / ||h_in||_F] по всем промптам.

    Это прокси для "насколько сильно повлияет ошибка сплита на этот слой":
    большой residual → слой много добавляет → ошибка сплита на этом слое накапливается быстрее.
    """
    text_model = model.model
    n_layers = len(text_model.layers)

    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def make_pre_hook(l):
        def hook(module, args):
            h = args[0] if isinstance(args, tuple) else args
            layer_inputs[l] = h.detach().float().cpu()
        return hook

    def make_post_hook(l):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_outputs[l] = h.detach().float().cpu()
        return hook

    for i, layer in enumerate(text_model.layers):
        hooks.append(layer.register_forward_pre_hook(make_pre_hook(i)))
        hooks.append(layer.register_forward_hook(make_post_hook(i)))

    per_layer_ratios = {l: [] for l in range(n_layers)}

    model.eval()
    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        layer_inputs.clear()
        layer_outputs.clear()
        with torch.no_grad():
            model(**inp)

        for l in range(n_layers):
            if l in layer_inputs and l in layer_outputs:
                h_in = layer_inputs[l]
                h_out = layer_outputs[l]
                ratio = frobenius_relative(h_out - h_in, h_in)
                per_layer_ratios[l].append(ratio)

    for h in hooks:
        h.remove()

    # Среднее по промптам
    return {l: sum(v) / len(v) for l, v in per_layer_ratios.items() if v}


def collect_head_contribution(
    model, tokenizer, prompts: list[str], device: str, n_gpu: int = 2
) -> dict[int, list[float]]:
    """
    Для каждого full_attention слоя: какой процент output energy вносит каждая группа голов?
    Возвращает: {layer_idx: [frac_group_0, frac_group_1, ...]}

    Это реальная оценка ошибки сплита: если group_0 даёт 60% energy,
    GPU_1 (которому не хватает group_0) потеряет 60% сигнала.
    """
    text_model = model.model
    layer_types = model.config.layer_types
    n_heads = model.config.num_attention_heads
    heads_per_gpu = n_heads // n_gpu

    head_contributions = {}

    # Хукаемся на attn output перед O-proj
    attn_outputs = {}
    hooks = []

    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    def make_attn_hook(l):
        def hook(module, input, output):
            # output[0] = attention hidden states [batch, seq, heads*head_dim]
            h = output[0] if isinstance(output, tuple) else output
            attn_outputs[l] = h.detach().float().cpu()
        return hook

    for l in full_attn_layers:
        layer = text_model.layers[l]
        # Хук на self-attention подмодуль
        if hasattr(layer, "self_attn"):
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))

    model.eval()
    for prompt in prompts[:4]:  # достаточно 4 промпта
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        attn_outputs.clear()
        with torch.no_grad():
            model(**inp)

        for l, h_full in attn_outputs.items():
            # h_full: [1, seq, heads*head_dim]
            # Разбиваем по группам голов
            head_dim = h_full.shape[-1] // n_heads
            group_norms = []
            for g in range(n_gpu):
                start = g * heads_per_gpu * head_dim
                end = (g + 1) * heads_per_gpu * head_dim
                group_h = h_full[..., start:end]
                group_norms.append(group_h.norm().item())

            total = sum(group_norms) + 1e-8
            fracs = [n / total for n in group_norms]

            if l not in head_contributions:
                head_contributions[l] = [[] for _ in range(n_gpu)]
            for g, frac in enumerate(fracs):
                head_contributions[l][g].append(frac)

    for h in hooks:
        h.remove()

    # Усредняем по промптам
    return {
        l: [sum(head_contributions[l][g]) / len(head_contributions[l][g]) for g in range(n_gpu)]
        for l in head_contributions
    }


def suggest_cca_points(
    residuals: dict[int, float],
    layer_types: list[str],
    budget: int = 8,
) -> list[int]:
    """
    Выбирает до `budget` точек CCA.
    Стратегия: сортируем слои по residual (убывание), берём топ-budget,
    потом размещаем CCA ПЕРЕД этими слоями (т.е. после l-1).
    """
    sorted_layers = sorted(residuals.items(), key=lambda x: x[1], reverse=True)
    top_layers = sorted([l for l, _ in sorted_layers[:budget]])
    return top_layers


def plot_results(
    residuals: dict[int, float],
    head_contribs: dict[int, list[float]],
    cca_data_driven: list[int],
    cca_uniform_k4: list[int],
    layer_types: list[str],
    output_path: Path,
):
    n_layers = len(residuals)
    layers = list(range(n_layers))
    res_vals = [residuals.get(l, 0) for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Residual magnitude
    colors = ["#c8e6c9" if layer_types[l] == "full_attention" else "#e3f2fd" for l in layers]
    bars = ax1.bar(layers, res_vals, color=colors, edgecolor="none", alpha=0.8)

    for p in cca_data_driven:
        ax1.axvline(p - 0.5, color="red", alpha=0.8, linewidth=2, linestyle="--",
                    label="CCA (data-driven)" if p == cca_data_driven[0] else "")
    for p in cca_uniform_k4:
        ax1.axvline(p - 0.5, color="orange", alpha=0.5, linewidth=1.5, linestyle=":",
                    label="CCA (K=4 uniform)" if p == cca_uniform_k4[0] else "")

    ax1.set_ylabel("Residual ratio ||Δh||/||h||")
    ax1.set_title("Qwen3.5-0.8B: Layer residual magnitude\n(green = full_attention, blue = linear_attention)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Head contribution balance (full attn only)
    full_attn_layers = sorted(head_contribs.keys())
    if full_attn_layers:
        fracs_g0 = [head_contribs[l][0] for l in full_attn_layers]
        fracs_g1 = [head_contribs[l][1] for l in full_attn_layers]
        x = range(len(full_attn_layers))
        ax2.bar(x, fracs_g0, label="GPU 0 heads", color="#4caf50", alpha=0.7)
        ax2.bar(x, fracs_g1, bottom=fracs_g0, label="GPU 1 heads", color="#f44336", alpha=0.7)
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([f"L{l}" for l in full_attn_layers])
        ax2.set_xlabel("Full attention layer")
        ax2.set_ylabel("Head energy fraction per GPU")
        ax2.set_title("Head contribution balance (N_gpu=2) — ideal=0.5/0.5")
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


def main():
    print(f"Device: {DEVICE}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Конфиг
    cfg = model.config
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    n_gpu = 2

    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    print(f"\nArchitecture: {n_layers} layers")
    print(f"Full attention at: {full_attn_layers}")
    print(f"n_heads={cfg.num_attention_heads}, n_kv_heads={cfg.num_key_value_heads}")
    print(f"linear_num_key_heads={cfg.linear_num_key_heads}")
    print(f"MTP layers (built-in lookahead): {cfg.mtp_num_hidden_layers}")

    # 1. Residual magnitudes
    print(f"\n[1/2] Collecting layer residuals ({len(PROMPTS)} prompts)...")
    t0 = time.time()
    residuals = collect_layer_residuals(model, tokenizer, PROMPTS, DEVICE)
    print(f"Done in {time.time() - t0:.1f}s")

    # 2. Head contribution
    print(f"\n[2/2] Collecting head contributions (full_attn layers, N_gpu={n_gpu})...")
    t0 = time.time()
    head_contribs = collect_head_contribution(model, tokenizer, PROMPTS, DEVICE, n_gpu=n_gpu)
    print(f"Done in {time.time() - t0:.1f}s")

    # CCA placement
    cca_data_driven = suggest_cca_points(residuals, layer_types, budget=8)
    cca_uniform_k4 = list(range(3, n_layers, 4))

    # Results
    print("\n=== RESIDUAL MAGNITUDE PER LAYER ===")
    for l in range(n_layers):
        lt = layer_types[l] if l < len(layer_types) else "?"
        full_mark = " [FULL]" if lt == "full_attention" else "       "
        cca_mark = " ← CCA (data-driven)" if l in cca_data_driven else ""
        print(f"  Layer {l:2d}{full_mark}: residual={residuals.get(l, 0):.4f}{cca_mark}")

    print("\n=== HEAD CONTRIBUTION BALANCE (full_attn, N_gpu=2) ===")
    print("  Ideal: 50% / 50%. Imbalance → larger split error for that layer.")
    for l in sorted(head_contribs.keys()):
        fracs = head_contribs[l]
        imbalance = abs(fracs[0] - 0.5)
        flag = " ⚠️  unbalanced" if imbalance > 0.05 else ""
        print(f"  Layer {l:2d}: GPU0={fracs[0]:.3f}  GPU1={fracs[1]:.3f}{flag}")

    print("\n=== CCA PLACEMENT ===")
    print(f"Data-driven (top-8 residual): {cca_data_driven}  ({len(cca_data_driven)} RTT/token)")
    print(f"Uniform K=4:                  {cca_uniform_k4}  ({len(cca_uniform_k4)} RTT/token)")
    print(f"\nLAN latency (5ms/RTT):")
    print(f"  Data-driven: {len(cca_data_driven)} × 5ms = {len(cca_data_driven)*5}ms/tok → {1000//(len(cca_data_driven)*5)} tok/s")
    print(f"  Uniform K=4: {len(cca_uniform_k4)} × 5ms = {len(cca_uniform_k4)*5}ms/tok → {1000//(len(cca_uniform_k4)*5)} tok/s")

    # Сохранить
    results = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "n_gpu": n_gpu,
        "layer_types": layer_types,
        "full_attention_layers": full_attn_layers,
        "residuals": {str(k): v for k, v in residuals.items()},
        "head_contributions": {str(k): v for k, v in head_contribs.items()},
        "cca_data_driven": cca_data_driven,
        "cca_uniform_k4": cca_uniform_k4,
    }
    results_path = RESULTS_DIR / "activation_profiling.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # График
    plot_path = RESULTS_DIR / "divergence_curve.png"
    plot_results(residuals, head_contribs, cca_data_driven, cca_uniform_k4, layer_types, plot_path)


if __name__ == "__main__":
    main()
