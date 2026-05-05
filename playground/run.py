#!/usr/bin/env python3
"""
TP-Surgical Playground — бенчмарк потенциального tok/s для любой конфигурации.

Использование:
  python playground/run.py --config playground/configs/qwen3_5_0.8b_lan2.yaml
  python playground/run.py --config playground/configs/qwen3_4b_lan2.yaml --latency 2.5
  python playground/run.py --list

Что симулируется:
  - Worker 0 видит col_0 attention heads (остальные зануляются хуком)
  - На каждой CCA-точке: time.sleep(RTT + transfer_overhead)
  - Измеряется реальный tok/s включая искусственную коммуникационную задержку
  - Строится детальный breakdown: compute / comm / per-CCA-point
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# ─── Добавляем корень проекта в PYTHONPATH ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from playground.sim import NetworkConfig, TPSurgicalSim, StepStats

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка конфига
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Вывод конфигурации
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          TP-Surgical Playground  ·  tok/s Simulator             ║
╚══════════════════════════════════════════════════════════════════╝"""


def print_config_summary(cfg: dict, net: NetworkConfig, device: str):
    m = cfg["model"]
    s = cfg["split"]
    b = cfg["benchmark"]
    cca = s["cca_layers"]
    print(BANNER)
    print(f"\n  Model       : {m['id']}")
    print(f"  Device      : {device}")
    print(f"  Split       : {s['n_gpu']} GPU, {len(cca)} CCA sync points → {cca}")
    print(f"  Network     : {net.topology}  latency={net.latency_one_way_ms}ms one-way"
          f"  RTT={net.rtt_ms}ms  BW={net.bandwidth_gbps}Gbps")
    print(f"  Benchmark   : prompt={b['prompt_len']} tok  "
          f"decode={b['n_decode_tokens']} tok  runs={b['n_runs']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Базовый baseline (без split)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_baseline(model, tokenizer, prompt_ids: torch.Tensor,
                     n_tokens: int, n_warmup: int, n_runs: int, device: str) -> float:
    """Tok/s полной модели без split/comm."""
    for _ in range(n_warmup):
        model.generate(prompt_ids, max_new_tokens=10,
                       do_sample=False, temperature=None, top_p=None)

    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(prompt_ids, max_new_tokens=n_tokens,
                       do_sample=False, temperature=None, top_p=None)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median = sorted(times)[len(times) // 2]
    return n_tokens / median


# ─────────────────────────────────────────────────────────────────────────────
# Симулированный TP-Surgical
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_tpsurgical(model, tokenizer, prompt_ids: torch.Tensor,
                       n_tokens: int, n_warmup: int, n_runs: int,
                       sim: TPSurgicalSim, device: str) -> tuple[float, list[StepStats]]:
    """
    Генерирует n_tokens токенов в режиме TP-Surgical.
    На каждом шаге декодирования вставляется задержка на CCA sync.
    """

    def _gen_with_sim(max_new: int) -> tuple[float, list[StepStats]]:
        input_ids = prompt_ids.clone()
        step_stats: list[StepStats] = []

        if device == "cuda":
            torch.cuda.synchronize()
        t_total = time.perf_counter()

        for _ in range(max_new):
            logits, st = sim.decode_step(model, input_ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            step_stats.append(st)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_total
        return elapsed, step_stats

    # warmup
    for _ in range(n_warmup):
        _gen_with_sim(5)

    all_times = []
    all_stats: list[StepStats] = []
    for _ in range(n_runs):
        elapsed, stats = _gen_with_sim(n_tokens)
        all_times.append(elapsed)
        all_stats = stats  # последний прогон

    median_elapsed = sorted(all_times)[len(all_times) // 2]
    tps = n_tokens / median_elapsed
    return tps, all_stats


# ─────────────────────────────────────────────────────────────────────────────
# Теоретический расчёт
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_tps(baseline_tps: float, net: NetworkConfig,
                    cca_layers: list[int], n_gpu: int,
                    hidden_size: int, dtype_bytes: int = 2) -> dict:
    """
    Оцениваем теоретический tok/s при идеальном параллелизме:
    - compute/token = 1/baseline_tps (один GPU)
    - split compute = compute/(n_gpu) (идеальное масштабирование)
    - comm = len(cca_layers) × sync_ms (hidden_size × dtype_bytes × 2 байт)
    """
    compute_ms = 1000 / baseline_tps
    split_compute_ms = compute_ms / n_gpu

    # Данные одного sync: [1, 1, H] bfloat16 × 2 направления
    sync_bytes = hidden_size * dtype_bytes * 2
    total_comm_ms = sum(net.sync_ms(sync_bytes) for _ in cca_layers)

    total_ms = split_compute_ms + total_comm_ms
    theory_tps = 1000 / total_ms if total_ms > 0 else float("inf")

    return {
        "compute_ms": compute_ms,
        "split_compute_ms": split_compute_ms,
        "comm_per_sync_ms": net.sync_ms(sync_bytes),
        "total_comm_ms": total_comm_ms,
        "total_ms": total_ms,
        "tps": theory_tps,
        "comm_ratio": total_comm_ms / total_ms if total_ms > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Отчёт
# ─────────────────────────────────────────────────────────────────────────────

def print_report(baseline_tps: float, sim_tps: float,
                 theory: dict, step_stats: list[StepStats],
                 net: NetworkConfig, cca_layers: list[int]):
    sep = "─" * 64

    print(f"\n{sep}")
    print(f"  РЕЗУЛЬТАТЫ")
    print(sep)

    # Speedup vs single GPU
    speedup = sim_tps / baseline_tps

    print(f"\n  {'Режим':<38} {'tok/s':>8}  {'vs single GPU':>14}")
    print(f"  {'─'*38} {'─'*8}  {'─'*14}")
    print(f"  {'Single GPU (no split)':<38} {baseline_tps:>8.1f}  {'—':>14}")
    print(f"  {'Теоретический TP-Surgical':<38} {theory['tps']:>8.1f}  "
          f"  {theory['tps']/baseline_tps:>+.2f}x")
    print(f"  {'Симулированный TP-Surgical':<38} {sim_tps:>8.1f}  "
          f"  {speedup:>+.2f}x")

    print(f"\n  {'─'*64}")
    print(f"  РАЗБИВКА (теоретическая, per token)")
    print(f"  {'─'*64}")
    print(f"  Compute (single GPU)        {theory['compute_ms']:>8.2f} ms")
    print(f"  Compute (split, {len(cca_layers[0:1])} GPU ideal) "
          f"         {theory['split_compute_ms']:>8.2f} ms")
    print(f"  Comm total ({len(cca_layers)} CCA × {theory['comm_per_sync_ms']:.2f}ms)"
          f"   {theory['total_comm_ms']:>8.2f} ms")
    print(f"  Total                       {theory['total_ms']:>8.2f} ms")
    print(f"  Comm ratio                  {theory['comm_ratio']*100:>7.1f}%")

    if step_stats:
        avg_compute = sum(s.compute_ms for s in step_stats) / len(step_stats)
        avg_comm    = sum(s.comm_ms    for s in step_stats) / len(step_stats)
        avg_total   = sum(s.total_ms   for s in step_stats) / len(step_stats)
        avg_cca     = sum(s.cca_hits   for s in step_stats) / len(step_stats)

        print(f"\n  {'─'*64}")
        print(f"  ИЗМЕРЕННЫЕ средние (последний прогон, {len(step_stats)} шагов)")
        print(f"  {'─'*64}")
        print(f"  Compute/token         {avg_compute:>8.2f} ms")
        print(f"  Comm/token            {avg_comm:>8.2f} ms  "
              f"({avg_comm/avg_total*100:.1f}%)")
        print(f"  Total/token           {avg_total:>8.2f} ms")
        print(f"  CCA hits/token        {avg_cca:>8.1f}")

    # Breakeven latency: при каком RTT sim_tps = baseline_tps
    compute_ms = theory["compute_ms"]
    split_ms   = theory["split_compute_ms"]
    n_cca      = len(cca_layers)
    if n_cca > 0:
        breakeven_rtt = (compute_ms - split_ms) / n_cca
        print(f"\n  Break-even RTT        {breakeven_rtt:>8.2f} ms  "
              f"(при этом RTT split = single GPU)")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

def latency_sweep(baseline_tps: float, net: NetworkConfig, cfg: dict):
    """Показывает ожидаемый tok/s при разных задержках сети."""
    s = cfg["split"]
    m = cfg["model"]
    cca_layers = s["cca_layers"]
    hidden_size = m["hidden_size"]
    n_gpu = s["n_gpu"]

    latencies = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    print(f"\n  {'─'*64}")
    print(f"  LATENCY SWEEP  (CCA points: {len(cca_layers)}, n_gpu: {n_gpu})")
    print(f"  {'─'*64}")
    print(f"  {'RTT (ms)':>10}  {'tok/s':>8}  {'vs baseline':>12}  {'comm%':>7}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*7}")

    for lat in latencies:
        net2 = NetworkConfig(net.topology, lat / 2, net.bandwidth_gbps)
        t = theoretical_tps(baseline_tps, net2, cca_layers, n_gpu,
                            hidden_size, dtype_bytes=2)
        marker = " ◄" if abs(lat - net.rtt_ms) < 0.01 else ""
        print(f"  {lat:>10.2f}  {t['tps']:>8.1f}  {t['tps']/baseline_tps:>+11.2f}x"
              f"  {t['comm_ratio']*100:>6.1f}%{marker}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CCA schedule sweep
# ─────────────────────────────────────────────────────────────────────────────

def cca_schedule_sweep(baseline_tps: float, net: NetworkConfig, cfg: dict):
    """Показывает как разное число CCA точек влияет на tok/s."""
    m = cfg["model"]
    s = cfg["split"]
    n_layers = m["n_layers"]
    hidden_size = m["hidden_size"]
    n_gpu = s["n_gpu"]
    current = s["cca_layers"]

    print(f"  {'─'*64}")
    print(f"  CCA SCHEDULE SWEEP  (RTT: {net.rtt_ms}ms, n_gpu: {n_gpu})")
    print(f"  {'─'*64}")
    print(f"  {'N CCA':>6}  {'tok/s':>8}  {'vs baseline':>12}  {'comm%':>7}  Layers")
    print(f"  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*7}  {'─'*20}")

    for k in range(1, min(n_layers + 1, 14)):
        step = n_layers // k
        layers = list(range(step - 1, n_layers, step))[:k]
        t = theoretical_tps(baseline_tps, net, layers, n_gpu, hidden_size)
        marker = " ◄" if layers == current else ""
        print(f"  {k:>6}  {t['tps']:>8.1f}  {t['tps']/baseline_tps:>+11.2f}x"
              f"  {t['comm_ratio']*100:>6.1f}%  {layers[:5]}"
              f"{'...' if len(layers) > 5 else ''}{marker}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def list_configs():
    configs_dir = Path(__file__).parent / "configs"
    print("\nДоступные конфиги:\n")
    for p in sorted(configs_dir.glob("*.yaml")):
        cfg = load_config(p)
        m = cfg["model"]
        s = cfg["split"]
        n = cfg["network"]
        print(f"  {p.name:<40} {m['id']}  {s['n_gpu']}GPU  {n['topology']}  "
              f"RTT={n['latency_one_way_ms']*2}ms")
    print()


def main():
    parser = argparse.ArgumentParser(description="TP-Surgical tok/s playground")
    parser.add_argument("--config", "-c", help="Путь к YAML конфигу")
    parser.add_argument("--list", "-l", action="store_true", help="Список конфигов")
    parser.add_argument("--latency", type=float, help="Переопределить one-way latency (мс)")
    parser.add_argument("--n-cca", type=int, help="Переопределить число CCA точек")
    parser.add_argument("--sweep", action="store_true",
                        help="Запустить latency sweep и CCA schedule sweep")
    parser.add_argument("--no-sim", action="store_true",
                        help="Только теория, без симуляции (быстрее)")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    if not args.config:
        parser.print_help()
        return

    cfg = load_config(args.config)
    m_cfg = cfg["model"]
    s_cfg = cfg["split"]
    b_cfg = cfg["benchmark"]
    n_cfg = cfg["network"]

    # CLI overrides
    if args.latency is not None:
        n_cfg["latency_one_way_ms"] = args.latency
    if args.n_cca is not None:
        n_layers = m_cfg["n_layers"]
        step = n_layers // args.n_cca
        s_cfg["cca_layers"] = list(range(step - 1, n_layers, step))[:args.n_cca]

    net = NetworkConfig(
        topology=n_cfg["topology"],
        latency_one_way_ms=n_cfg["latency_one_way_ms"],
        bandwidth_gbps=n_cfg["bandwidth_gbps"],
    )

    device = resolve_device()
    print_config_summary(cfg, net, device)

    # Загружаем модель
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Загружаем {m_cfg['id']} ...")
    tokenizer = AutoTokenizer.from_pretrained(m_cfg["id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["id"], dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    print(f"  Параметров: {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    # Реальные параметры из модели (если в конфиге null)
    real_cfg = model.config
    if s_cfg.get("full_attn_layers") is None:
        # Все слои — standard transformer
        s_cfg["full_attn_layers"] = list(range(m_cfg["n_layers"]))

    # Prompt
    prompt = "Explain in detail how neural networks learn from data."
    prompt_ids = tokenizer(
        prompt, return_tensors="pt",
        max_length=b_cfg["prompt_len"], truncation=True
    )["input_ids"].to(device)

    # 1. Baseline
    print("  Измеряем baseline (single GPU)...")
    baseline_tps = measure_baseline(
        model, tokenizer, prompt_ids,
        n_tokens=b_cfg["n_decode_tokens"],
        n_warmup=b_cfg["n_warmup"],
        n_runs=b_cfg["n_runs"],
        device=device,
    )
    print(f"  Baseline: {baseline_tps:.1f} tok/s\n")

    # 2. Теоретический расчёт
    theory = theoretical_tps(
        baseline_tps, net,
        cca_layers=s_cfg["cca_layers"],
        n_gpu=s_cfg["n_gpu"],
        hidden_size=m_cfg["hidden_size"],
    )

    # 3. Симуляция
    sim_tps = 0.0
    step_stats: list[StepStats] = []

    if not args.no_sim:
        sim = TPSurgicalSim(
            model=model,
            net=net,
            full_attn_layers=s_cfg["full_attn_layers"],
            cca_layers=s_cfg["cca_layers"],
            n_gpu=s_cfg["n_gpu"],
            device=device,
        )

        print("  Симулируем TP-Surgical (с искусственными задержками)...")
        sim_tps, step_stats = measure_tpsurgical(
            model, tokenizer, prompt_ids,
            n_tokens=b_cfg["n_decode_tokens"],
            n_warmup=b_cfg["n_warmup"],
            n_runs=b_cfg["n_runs"],
            sim=sim,
            device=device,
        )
        print(f"  Симулированный: {sim_tps:.1f} tok/s\n")

    # 4. Отчёт
    print_report(baseline_tps, sim_tps or theory["tps"], theory,
                 step_stats, net, s_cfg["cca_layers"])

    # 5. Sweep
    if args.sweep:
        latency_sweep(baseline_tps, net, cfg)
        cca_schedule_sweep(baseline_tps, net, cfg)


if __name__ == "__main__":
    main()
