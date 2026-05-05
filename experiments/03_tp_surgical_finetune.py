"""
TP-Surgical: KL-distillation pipeline. Целевое железо — H100.

Оба пула из OpenHermes-2.5, никакой генерации.
Teacher logits вычисляются on-the-fly (кэш logits убран — при 50K примерах
он весил бы ~4 TB из-за vocab_size=248K).

Структура кэша:
  cache/s1_pool.pt    — токены S1 из OpenHermes[0:N_POOL]
  cache/s2_pool.pt    — токены S2 из OpenHermes[N_POOL:2*N_POOL]
  results/cca_s1.pt      — CCA веса (финал S1)
  results/cca_s1_best.pt — лучшие CCA веса по held-out PPL
  results/cca_s2.pt      — CCA веса (финал S2)
  results/cca_s2_best.pt — лучшие CCA веса S2
  results/model_s2.pt    — model state_dict (финал S2)
  results/tp_surgical_train.json

Данные: experiments/data/openhermes_100k
Запуск: python experiments/03_tp_surgical_finetune.py
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

OPENHERMES_PATH = Path("experiments/data/openhermes_100k")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
CCA_LAYERS       = [0, 3, 7, 11, 15, 19, 23]

# ── Гиперпараметры ────────────────────────────────────────────
# Целевое железо: H100 80GB  (~25ms/step → S1 200K шагов ≈ 1.4ч, S2 50K ≈ 25мин)
N_POOL     = 50000  # примеров из OpenHermes для каждого пула (100K суммарно)
MAX_SEQ    = 128
N_STEPS_S1 = 200000 # S1: только CCA, frozen base
N_STEPS_S2 = 50000  # S2: CCA + unfrozen base (tiny LR)
LR_CCA     = 5e-5   # conservative — 50K уникальных примеров, overfit маловероятен
LR_BASE    = 5e-7   # base model почти не трогаем
TEMP_KL    = 3.0
LOG_EVERY  = 2000
EVAL_EVERY = 10000

CACHE_DIR   = Path("experiments/cache")
RESULTS_DIR = Path("experiments/results")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Модель: CCACorrection + Simulator
# ─────────────────────────────────────────────────────────────

class CCACorrection(nn.Module):
    """
    SwiGLU correction с learnable scale.
    Intermediate dim = H//2 (5.5M total для 7 модулей вместо 44M).
    scale стартует в 0 → sigmoid(0)=0.5, но down.weight=0 → выход=0 при старте.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = hidden_size // 2
        self.norm  = nn.RMSNorm(hidden_size, elementwise_affine=True)
        self.gate  = nn.Linear(hidden_size, mid, bias=False)
        self.up    = nn.Linear(hidden_size, mid, bias=False)
        self.down  = nn.Linear(mid, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0.5; down=0 → identity
        nn.init.zeros_(self.down.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x   = self.norm(h)
        cor = self.down(F.silu(self.gate(x)) * self.up(x))
        return h + torch.sigmoid(self.scale) * cor


class TPSurgicalSimulator:
    """
    Контекст-менеджер: маскирует col_1 attention heads перед O-proj
    и применяет CCA correction после CCA слоёв.
    """
    def __init__(self, model, cca: nn.ModuleDict, n_gpu: int = 2):
        self.model  = model
        self.cca    = cca
        self.n_gpu  = n_gpu
        self._hooks: list = []

    def _install(self):
        cfg      = self.model.config
        col_size = (cfg.num_attention_heads // self.n_gpu) * cfg.head_dim

        for l_idx, layer in enumerate(self.model.model.layers):
            # Head mask на o_proj входе
            if l_idx in FULL_ATTN_LAYERS:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                    def _mask(module, args, cs=col_size):
                        inp = args[0]
                        m   = torch.ones_like(inp)
                        m[..., cs:] = 0.0
                        return (inp * m,)
                    self._hooks.append(
                        layer.self_attn.o_proj.register_forward_pre_hook(_mask)
                    )
            # CCA correction на выходе слоя
            if l_idx in CCA_LAYERS and str(l_idx) in self.cca:
                corr = self.cca[str(l_idx)]
                def _cca(module, inp, out, c=corr):
                    h  = out[0] if isinstance(out, tuple) else out
                    h2 = c(h)
                    return (h2,) + out[1:] if isinstance(out, tuple) else h2
                self._hooks.append(layer.register_forward_hook(_cca))

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, *_):
        self._remove()


# ─────────────────────────────────────────────────────────────
# Distillation step
# ─────────────────────────────────────────────────────────────

def kl_distill_step(model, sim, ids: torch.Tensor,
                    temperature: float = TEMP_KL,
                    teacher_logits_cached: torch.Tensor | None = None):
    """
    KL(teacher || student) + 0.3 * hard_LM(student).
    teacher_logits_cached: опциональный pre-computed teacher (для S2).
    """
    ids = ids.to(DEVICE)  # гарантируем правильное устройство
    if teacher_logits_cached is not None:
        teacher_logits = teacher_logits_cached.to(DEVICE)
    else:
        with torch.no_grad():
            teacher_logits = model(input_ids=ids).logits.float()

    with sim:
        student_logits = model(input_ids=ids).logits.float()

    T_l    = teacher_logits[:, :-1, :].contiguous()
    S_l    = student_logits[:, :-1, :].contiguous()
    labels = ids[:, 1:].contiguous()

    P     = F.softmax(T_l / temperature, dim=-1)
    log_Q = F.log_softmax(S_l / temperature, dim=-1)
    kl    = F.kl_div(log_Q, P, reduction="batchmean") * (temperature ** 2)
    hard  = F.cross_entropy(S_l.view(-1, S_l.size(-1)), labels.view(-1))

    return 0.7 * kl + 0.3 * hard, kl.item(), hard.item()


# ─────────────────────────────────────────────────────────────
# Пул данных из OpenHermes
# ─────────────────────────────────────────────────────────────

def load_or_tokenize_pool(tokenizer, n_seqs: int, offset: int,
                          cache_path: Path, label: str) -> list:
    """
    Загружает OpenHermes[offset : offset+n_seqs], форматирует разговоры
    как 'Human: ...\nAssistant: ...', токенизирует до MAX_SEQ.
    Результат кэшируется в cache_path.
    """
    if cache_path.exists():
        print(f"  [{label}] Loading from cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [{label}] {len(data)} sequences")
        return data

    if not OPENHERMES_PATH.exists():
        raise FileNotFoundError(
            f"OpenHermes не найден: {OPENHERMES_PATH}\n"
            "Скачай: python -c \"from datasets import load_dataset, Dataset; "
            "ds=load_dataset('teknium/OpenHermes-2.5',split='train',streaming=True); "
            "examples=[e for i,e in zip(range(10000),ds)]; "
            "Dataset.from_list(examples).save_to_disk('experiments/data/openhermes_10k')\""
        )

    from datasets import load_from_disk
    ds = load_from_disk(str(OPENHERMES_PATH))

    pool = []
    for ex in ds.select(range(offset, min(offset + n_seqs * 2, len(ds)))):
        convs = ex.get("conversations", [])
        parts = []
        for turn in convs:
            role = "Human" if turn["from"] == "human" else "Assistant"
            parts.append(f"{role}: {turn['value'].strip()}")
        text = "\n".join(parts)
        if len(text) < 20:
            continue
        ids = tokenizer(
            text, return_tensors="pt",
            max_length=MAX_SEQ, truncation=True, add_special_tokens=True,
        )["input_ids"]
        if ids.shape[1] >= 8:
            pool.append(ids)
        if len(pool) >= n_seqs:
            break

    print(f"  [{label}] {len(pool)} sequences from OpenHermes[{offset}:{offset+n_seqs}]")
    torch.save(pool, cache_path)
    return pool


# teacher logits вычисляются on-the-fly в kl_distill_step —
# кэш убран: при vocab_size=248K и 50K примерах он занял бы ~4 TB.


# ─────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────

HELD_TEXTS = [
    "Convolutional networks slide learned filters across inputs, detecting local patterns regardless of position. MaxPooling reduces spatial dimensions while retaining salient features.",
    "The softmax function converts logits into probabilities by exponentiating and normalizing. Subtracting the max logit prevents numerical overflow.",
    "Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B). It updates the prior belief about A given observed evidence B.",
    "Gradient clipping prevents exploding gradients by rescaling gradient norms that exceed a threshold value.",
    "A compiler transforms source code through lexing, parsing, semantic analysis, optimization, and code generation.",
]

EVAL_PROMPTS = [
    "Explain how attention mechanisms work in neural networks.",
    "Write a Python function that checks if a number is prime.",
    "What is the difference between RAM and ROM?",
    "How does TCP ensure reliable data delivery?",
]


@torch.no_grad()
def eval_ppl(model, tokenizer, texts, sim=None, device=DEVICE, n=5):
    model.eval()
    total = 0.0
    for t in texts[:n]:
        ids = tokenizer(t, return_tensors="pt",
                        max_length=MAX_SEQ, truncation=True)["input_ids"].to(device)
        if sim:
            with sim:
                logits = model(input_ids=ids).logits.float()
        else:
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        total += loss.item()
    return float(torch.exp(torch.tensor(total / n)))


@torch.no_grad()
def generate_compare(model, tokenizer, prompts, sim, device, max_new=100):
    model.eval()
    rows = []
    for prompt in prompts:
        inp  = tokenizer(prompt, return_tensors="pt").to(device)
        n_in = inp["input_ids"].shape[1]

        t0    = time.time()
        out_f = model.generate(**inp, max_new_tokens=max_new,
                               do_sample=False, temperature=None, top_p=None)
        tps_f = max_new / (time.time() - t0)

        t0 = time.time()
        with sim:
            out_s = model.generate(**inp, max_new_tokens=max_new,
                                   do_sample=False, temperature=None, top_p=None)
        tps_s = max_new / (time.time() - t0)

        rows.append({
            "prompt":    prompt,
            "full":      tokenizer.decode(out_f[0][n_in:], skip_special_tokens=True),
            "full_tps":  round(tps_f, 1),
            "split":     tokenizer.decode(out_s[0][n_in:], skip_special_tokens=True),
            "split_tps": round(tps_s, 1),
        })
    return rows


def print_table(rows, ppl_full):
    print(f"\n{'Режим':<30} {'Held PPL':>10}  {'vs full':>8}")
    print("─" * 52)
    for name, ppl in rows:
        sign = "+" if ppl > ppl_full else ""
        vs   = f"{sign}{(ppl/ppl_full-1)*100:.1f}%"
        print(f"{name:<30} {ppl:>10.2f}  {vs:>8}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    hidden_size = model.config.hidden_size  # 1024
    cca = nn.ModuleDict({
        str(l): CCACorrection(hidden_size).to(DEVICE).to(torch.bfloat16)
        for l in CCA_LAYERS
    })
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M  "
          f"CCA: {sum(p.numel() for p in cca.parameters())/1e6:.1f}M")

    # Если есть сохранённые CCA S1 — грузим сразу
    if (RESULTS_DIR / "cca_s1.pt").exists():
        print("[cache] Loading CCA S1 weights...")
        sd = torch.load(RESULTS_DIR / "cca_s1.pt", map_location=DEVICE, weights_only=True)
        for k, v in cca.items():
            if k in sd:
                v.load_state_dict(sd[k])

    sim = TPSurgicalSimulator(model, cca)

    # ── Baseline ──────────────────────────────────────────────
    ppl_full   = eval_ppl(model, tokenizer, HELD_TEXTS)
    ppl_split0 = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
    print(f"\nBaseline:  Full={ppl_full:.2f}  Split={ppl_split0:.2f}  "
          f"(+{(ppl_split0/ppl_full-1)*100:.1f}%)")

    # ─────────────────────────────────────────────────────────
    # STAGE 1: factual self-generated data, CCA only
    # ─────────────────────────────────────────────────────────
    skip_s1 = (RESULTS_DIR / "cca_s1.pt").exists()
    if skip_s1:
        print("\n[cache] Stage 1 already done (cca_s1.pt exists), loading results...")
        # PPL уже с загруженными весами
        ppl_s1 = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
        r1     = (ppl_split0 - ppl_s1) / (ppl_split0 - ppl_full) * 100
        print(f"S1 (cached):  held PPL={ppl_s1:.2f}  recovery={r1:.0f}%")
        log_s1 = []
    else:
        print("\n" + "═"*60)
        print(f"STAGE 1: OpenHermes distillation ({N_STEPS_S1} steps, CCA only)")
        print("═"*60)

        s1_pool = load_or_tokenize_pool(
            tokenizer, N_POOL, offset=0,
            cache_path=CACHE_DIR / "s1_pool.pt", label="S1",
        )

        for p in model.parameters():
            p.requires_grad_(False)
        for p in cca.parameters():
            p.requires_grad_(True)
        cca.train()

        opt1   = AdamW(cca.parameters(), lr=LR_CCA, weight_decay=0.01)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=N_STEPS_S1, eta_min=LR_CCA * 0.1
        )
        log_s1 = []
        t0     = time.time()

        best_ppl_s1 = float("inf")
        for step in range(N_STEPS_S1):
            ids = s1_pool[step % len(s1_pool)]
            opt1.zero_grad()
            loss, kl, hard = kl_distill_step(model, sim, ids, TEMP_KL)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cca.parameters(), 0.5)
            opt1.step()
            sched1.step()
            log_s1.append({"kl": kl, "hard": hard})
            if (step + 1) % LOG_EVERY == 0:
                avg_kl   = sum(x["kl"]   for x in log_s1[-LOG_EVERY:]) / LOG_EVERY
                avg_hard = sum(x["hard"] for x in log_s1[-LOG_EVERY:]) / LOG_EVERY
                print(f"  step {step+1:4d}  KL={avg_kl:.4f}  hard={avg_hard:.4f}"
                      f"  LR={sched1.get_last_lr()[0]:.2e}", flush=True)
            if (step + 1) % EVAL_EVERY == 0:
                cca.eval()
                cur_ppl = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
                cca.train()
                r_cur = (ppl_split0 - cur_ppl) / (ppl_split0 - ppl_full) * 100
                print(f"  [eval] step {step+1}  held PPL={cur_ppl:.2f}  recovery={r_cur:.0f}%", flush=True)
                if cur_ppl < best_ppl_s1:
                    best_ppl_s1 = cur_ppl
                    torch.save({k: v.state_dict() for k, v in cca.items()},
                               RESULTS_DIR / "cca_s1_best.pt")

        cca.eval()
        ppl_s1 = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
        r1     = (ppl_split0 - ppl_s1) / (ppl_split0 - ppl_full) * 100
        print(f"S1 done ({time.time()-t0:.0f}s):  held PPL={ppl_s1:.2f}  recovery={r1:.0f}%")

        # Сохраняем CCA S1
        torch.save({k: v.state_dict() for k, v in cca.items()},
                   RESULTS_DIR / "cca_s1.pt")
        print(f"  → saved: {RESULTS_DIR / 'cca_s1.pt'}")

    # ─────────────────────────────────────────────────────────
    # STAGE 2: instruction-style data, CCA + base unfreeze
    # ─────────────────────────────────────────────────────────
    skip_s2 = (RESULTS_DIR / "cca_s2.pt").exists()
    if skip_s2:
        print("\n[cache] Stage 2 already done (cca_s2.pt exists), loading...")
        sd2 = torch.load(RESULTS_DIR / "cca_s2.pt", map_location=DEVICE, weights_only=True)
        for k, v in cca.items():
            if k in sd2:
                v.load_state_dict(sd2[k])
        if (RESULTS_DIR / "model_s2.pt").exists():
            print("[cache] Loading model_s2.pt...")
            model.load_state_dict(
                torch.load(RESULTS_DIR / "model_s2.pt", map_location=DEVICE, weights_only=True)
            )
        ppl_s2 = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
        r2     = (ppl_split0 - ppl_s2) / (ppl_split0 - ppl_full) * 100
        print(f"S2 (cached):  held PPL={ppl_s2:.2f}  recovery={r2:.0f}%")
        log_s2 = []
    else:
        print("\n" + "═"*60)
        print(f"STAGE 2: OpenHermes distillation ({N_STEPS_S2} steps, CCA + base)")
        print("═"*60)

        s2_pool = load_or_tokenize_pool(
            tokenizer, N_POOL, offset=N_POOL,
            cache_path=CACHE_DIR / "s2_pool.pt", label="S2",
        )

        for p in model.parameters():
            p.requires_grad_(True)
        cca.train()
        model.train()

        opt2   = AdamW([
            {"params": cca.parameters(),   "lr": LR_CCA * 0.5},
            {"params": model.parameters(), "lr": LR_BASE},
        ], weight_decay=0.01)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=N_STEPS_S2)

        log_s2 = []
        t0     = time.time()

        best_ppl_s2 = float("inf")
        for step in range(N_STEPS_S2):
            idx       = step % len(s2_pool)
            ids       = s2_pool[idx]
            teacher_l = teacher_cache[idx]

            opt2.zero_grad()
            loss, kl, hard = kl_distill_step(
                model, sim, ids, TEMP_KL, teacher_logits_cached=teacher_l
            )
            loss.backward()
            all_p = list(cca.parameters()) + list(model.parameters())
            torch.nn.utils.clip_grad_norm_(all_p, 0.5)
            opt2.step()
            sched2.step()
            log_s2.append({"kl": kl, "hard": hard})
            if (step + 1) % LOG_EVERY == 0:
                avg_kl   = sum(x["kl"]   for x in log_s2[-LOG_EVERY:]) / LOG_EVERY
                avg_hard = sum(x["hard"] for x in log_s2[-LOG_EVERY:]) / LOG_EVERY
                print(f"  step {step+1:4d}  KL={avg_kl:.4f}  hard={avg_hard:.4f}", flush=True)
            if (step + 1) % EVAL_EVERY == 0:
                model.eval(); cca.eval()
                cur_ppl = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
                model.train(); cca.train()
                r_cur = (ppl_split0 - cur_ppl) / (ppl_split0 - ppl_full) * 100
                print(f"  [eval] step {step+1}  held PPL={cur_ppl:.2f}  recovery={r_cur:.0f}%", flush=True)
                if cur_ppl < best_ppl_s2:
                    best_ppl_s2 = cur_ppl
                    torch.save({k: v.state_dict() for k, v in cca.items()},
                               RESULTS_DIR / "cca_s2_best.pt")

        model.eval()
        cca.eval()
        ppl_s2 = eval_ppl(model, tokenizer, HELD_TEXTS, sim=sim)
        r2     = (ppl_split0 - ppl_s2) / (ppl_split0 - ppl_full) * 100
        print(f"S2 done ({time.time()-t0:.0f}s):  held PPL={ppl_s2:.2f}  recovery={r2:.0f}%")

        # Сохраняем CCA S2 + model
        torch.save({k: v.state_dict() for k, v in cca.items()},
                   RESULTS_DIR / "cca_s2.pt")
        torch.save(model.state_dict(), RESULTS_DIR / "model_s2.pt")
        print(f"  → saved: {RESULTS_DIR / 'cca_s2.pt'}, {RESULTS_DIR / 'model_s2.pt'}")

    # ── Итоги ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("ИТОГИ (held-out)")
    print("═"*60)
    print_table([
        ("Full model",          ppl_full),
        ("Split, no CCA",       ppl_split0),
        ("Split + CCA Stage 1", ppl_s1),
        ("Split + CCA Stage 2", ppl_s2),
    ], ppl_full)

    # ── Генерация ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("ГЕНЕРАЦИЯ")
    print("═"*60)
    gens = generate_compare(model, tokenizer, EVAL_PROMPTS, sim, DEVICE)
    for g in gens:
        print(f"\n[Prompt] {g['prompt']}")
        print(f"[Full   | {g['full_tps']:.1f} tok/s]\n  {g['full'][:300]}")
        print(f"[Split  | {g['split_tps']:.1f} tok/s]\n  {g['split'][:300]}")

    # ── Финальный JSON ────────────────────────────────────────
    result = {
        "model": MODEL_ID, "cca_layers": CCA_LAYERS,
        "n_gen_seqs": N_GEN_SEQS,
        "ppl_full":          ppl_full,
        "ppl_split_untrained": ppl_split0,
        "s1": {"held_ppl": ppl_s1, "recovery_pct": r1, "steps": N_STEPS_S1},
        "s2": {"held_ppl": ppl_s2, "recovery_pct": r2, "steps": N_STEPS_S2},
        "generations": gens,
    }
    out_path = RESULTS_DIR / "tp_surgical_train.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
