---
title: "Diffusion-State Transformer (DST) — Research Spec"
date: 2026-04-28
author: TBD
version: v1.0
status: research-draft
stream: wan
tags: [DST, diffusion, capsule, research, phase-0, wan]
---

# Diffusion-State Transformer (DST) — Research Spec

**Статус:** research draft, Phase 0. Цель документа — зафиксировать формализацию, архитектуру, потери, план экспериментов и точки риска так, чтобы 0.1B прототип можно было реализовать без новых архитектурных решений.

**Главная гипотеза.** Capsule state можно моделировать как точку на манифольде "валидных состояний" в continuous-time noise hierarchy. Тогда failover, jitter, corruption, и tier escalation становятся одной осью τ ∈ [0, T_max], а не разными корректирующими механизмами.

Если эта гипотеза не выдержит на 0.1B, fallback — стандартная capsule arch с adversarial augmentation (см. `../architecture/columnar-decentralized-llm.md` §8).

---

## 1. Формализация

### 1.1 Обозначения

- `c ∈ R^(N×D)` — capsule: N memory slots размерности D.
- `x_{1..t}` — токены до текущего шага.
- `w_t` — sliding window (последние W токенов).
- `τ ∈ [0, T_max]` — noise level / "staleness". Скаляр, condition на модель.
- `c_τ` — наблюдаемая (зашумлённая) capsule.
- `c_0` — "clean" capsule, target denoiser.

### 1.2 Forward (noising) процесс

Variance-preserving SDE (как EDM, Karras et al):

```
c_τ = α(τ) · c_0 + σ(τ) · ε,   ε ~ N(0, I)
α(τ)² + σ(τ)² = 1
```

Используем EDM-параметризацию:
- `σ(τ) = τ` (linear noise scale)
- `T_max = 80` (пустая Gaussian, при которой капсула полностью разрушена)
- Sampling τ во время тренировки: `log τ ~ N(P_mean=−1.2, P_std=1.2)`, как в EDM

### 1.3 Reverse (denoising) процесс

Обучаем denoiser `D_θ(c_τ, τ, w_t) → ĉ_0`. На inference используем **Consistency Training** parameterization (Song et al, 2023):

```
D_θ(c_τ, τ, w) = c_skip(τ) · c_τ + c_out(τ) · F_θ(c_τ, τ, w)
```

где `c_skip` и `c_out` — preconditioning, гарантирующие что D(c_0, 0) = c_0 by construction.

**Inference path:** 1 denoising step. Если τ велико и compute позволяет — 2-3 шага через consistency multistep, но это optional.

### 1.4 Маппинг τ ↔ runtime сигналы

τ — внутренняя ось модели. Runtime вычисляет `τ_observed` из:

```
τ_observed = clip(
    a · seconds_since_checkpoint
  + b · transmission_compression_ratio
  + c · packet_loss_indicator
  + d · failover_recency
  , 0, T_max
)
```

Коэффициенты a, b, c, d калибруются пост-hoc на validation set: подбираются так, чтобы predicted τ соответствовал тому τ, при котором модель показывает наблюдаемую quality degradation.

### 1.5 State update

Capsule эволюционирует:

```
c_{t+1} = U_θ(D_θ(c_τ_t, τ_t, w_t), w_t, x_t)
```

То есть: сначала denoise → потом state update на чистой версии. **Никогда** не делаем state update на noisy capsule напрямую — это критично для математики тренировки (см. §3.4).

---

## 2. Архитектура

### 2.1 Capsule layout (Phase 0)

| Параметр | Значение |
| --- | ---: |
| N (slots) | 32 |
| D (slot dim) | 256 |
| Размер capsule fp16 | 16 KB |
| Sliding window W | 64 tokens |

Phase 1 (0.5B): N=64, D=384, ~48 KB.
Phase 2 (7B): N=128, D=512, ~128 KB до сжатия. Целевой transmit ≤ 5 MB после сжатия checkpoint chain (см. §4.4).

### 2.2 Блоки

Backbone — стандартный pre-norm transformer на conjoined sequence `[slots; window]`. Не decoder-only; capsule slots видят window и наоборот через single bidirectional attention внутри блока.

```
Block:
  norm → attention([slots; window]) → +residual
  norm → MLP → +residual
  + AdaLN(τ) conditioning   ← добавляется к каждому norm как масштаб/сдвиг
```

`AdaLN(τ)` — sinusoidal embedding(τ) → MLP → (γ, β) для каждого LayerNorm. Это стандарт из DiT (Peebles & Xie).

### 2.3 Раздельные головы

Из общего backbone выходят:

| Голова | Вход | Выход | Функция |
|---|---|---|---|
| `D_θ` (denoiser) | last-block(slots) | ĉ_0 (N×D) | denoise capsule |
| `U_θ` (state update) | last-block(slots), window, new tokens | c_{t+1} | gated update slots |
| `LM_head` | last-block(window[-1]) | logits over vocab | next token |
| `Lookahead_k` (k=1..3) | last-block(window[-1]) | logits | предсказание t+k+1 |
| `Struct_head` | last-block(window[-1]) | logits over coarse classes | hierarchical lookahead (см. §2.4) |
| `Uncertainty_head` | last-block(window[-1]) | scalar | self-uncertainty signal |

D_θ и U_θ делят 8 нижних блоков, специализируются на 2 верхних. LM_head, Lookahead_k и Struct_head делят backbone полностью, разные linear projections.

### 2.4 Hierarchical lookahead

```
Struct_head:    предсказывает coarse class токена t+1 (1 of 64 кластеров BPE по embedding)
Lookahead_1:    next token t+1
Lookahead_2:    t+2 conditioned on t+1
Lookahead_3:    t+3 conditioned on t+2
```

Acceptance каскад: при verification gateway сначала проверяет coarse class match — если miss, lookahead отбрасывается без пересчёта tokens.

Параметрически условим все lookahead heads на τ: при высоком τ модель сама выдаёт меньший effective chunk size (через learned gating, не hard cutoff).

---

## 3. Тренировка

### 3.1 Полный список потерь

Пусть `c_0` — clean capsule, полученная teacher-forcing rollout-ом без шума на текущем sample. `τ ~ p(τ)`, `c_τ = α(τ)c_0 + σ(τ)ε`.

```
L_LM            = CE(LM_head(D_θ(c_τ, τ, w_t)), x_t)
L_CT            = ||D_θ(c_τ, τ, w) − stop_grad(D_θ_EMA(c_τ', τ', w))||²
                  где (τ', τ) — соседние точки на noise schedule
L_state_consis  = ||U_θ(D_θ(c_τ, τ, w), w, x_t) − stop_grad(c_0_next)||²
L_lookahead     = Σ_k λ_k · CE(Lookahead_k(D_θ(c_τ, τ, w)), x_{t+k})
L_struct        = CE(Struct_head(D_θ(c_τ, τ, w)), class(x_{t+1}))
L_failover      = CE(LM_head через synthetic failover trajectory, x_t)
                  (см. §3.3)
L_uncertainty   = MSE(Uncertainty_head, |true_KL(p_θ || p_oracle)|)
                  только когда oracle доступен (5% батчей)
```

Итог:
```
L_total = w_LM·L_LM + w_CT·L_CT + w_SC·L_state_consis
        + w_LA·L_lookahead + w_ST·L_struct + w_FO·L_failover
        + w_U·L_uncertainty
```

Стартовые веса: `(1.0, 0.3, 0.5, 0.5, 0.2, 0.5, 0.1)`. Все ≥ 0.

### 3.2 Curriculum

Phase 0 training в три этапа:

| Этап | Ходы | Включенные потери | τ диапазон |
|---|---:|---|---|
| A | 0–30% | L_LM only, τ=0 forced | {0} |
| B | 30–70% | + L_state_consis, + L_lookahead, + L_struct | log-uniform на [0, 1] |
| C | 70–100% | все потери | log-uniform на [0, T_max=80] |

Цель этапа A — получить разумный capsule baseline до того, как добавляется diffusion. Этап C доводит модель до робастности при экстремальном τ.

### 3.3 Synthetic failover

Failover генерируется так:
1. Берём rollout длиной T токенов с teacher-forced clean capsule c_0,1..T.
2. Случайно выбираем `t_fail ∈ [W, T)` и `Δ ∈ [10, 100]` (gap token count).
3. Подаём на вход `c_observed = α(τ_fail) · c_0,t_fail-Δ + σ(τ_fail) · ε`, имитируя "новый воркер получил stale capsule на Δ токенов раньше + noise".
4. Заставляем модель сгенерировать токены `[t_fail .. T]`, считаем CE.

τ_fail сэмплируется из верхней половины τ-распределения: failover — это всегда высокий шум.

### 3.4 Почему важен stop_grad

`L_state_consis` — критично, чтобы denoise + update сходились к той же траектории, что чистый rollout. Но если разрешить градиент течь через `c_0_next` (вычисленный rollout-ом), мы получим bootstrap loop, в котором модель учится на собственных предсказаниях. Это разрушает обучение (mode collapse на тривиальный update).

Решение: clean rollout вычисляется **под no_grad**, его результаты служат target. Это похоже на DAgger / teacher forcing для recurrent models.

### 3.5 Гиперпараметры (Phase 0)

| Параметр | Значение |
| --- | ---: |
| Model size | ~120M params |
| Layers | 12 |
| Hidden | 512 |
| Heads | 8 |
| Vocab | 32K (BPE) |
| Context (window+slots) | 64 + 32 = 96 tokens |
| Batch | 256 sequences × 1024 tokens |
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.1 |
| LR schedule | warmup 2K → cosine to 1e-5 |
| Peak LR | 6e-4 |
| Total tokens | 30B (TinyStories + RedPajama сэмпл) |
| EMA decay | 0.9999 (для CT target) |
| Train compute | ~8× A100-80GB · 5 дней |

### 3.6 Данные

- **TinyStories** (Phase 0 prototype). Помогает отделить language quality от capsule quality.
- **RedPajama-v2 sample**, 5B токенов, для general LM сигнала.
- **Pseudo-conversations:** склейка 2-8 коротких текстов с разделителями для имитации multi-turn.
- **Augmentation:** при формировании batch случайно вставляем "checkpoint events" (record c_0 → discard → resume from c_0+noise) для тренировки failover.

---

## 4. Inference

### 4.1 Worker decoding step

```python
def decode_step(c_observed, τ, window, new_input):
    # Step 1: denoise (1-step consistency)
    c_clean_hat = D_theta(c_observed, τ, window)

    # Step 2: state update with new content
    c_new = U_theta(c_clean_hat, window, new_input)

    # Step 3: token generation
    logits = LM_head(window, c_new)
    token = sample(logits)

    # Step 4: lookahead chunk (hierarchical)
    chunk = lookahead_chain(window + [token], c_new, τ_aware=True)

    # Step 5: uncertainty
    u = Uncertainty_head(window, c_new)

    return token, chunk, c_new, u
```

### 4.2 Когда денойзить

Денойзинг — это лишний forward через D_θ. Чтобы не платить per-token:
- Если `τ < τ_lo` (≈ 0.1): пропустить denoise, использовать `c_observed` напрямую.
- Если `τ_lo ≤ τ < τ_hi` (≈ 1.0): 1-step denoise.
- Если `τ ≥ τ_hi`: 2-step consistency (multistep).

Это поднимает amortized cost: ~5-10% additional forward time на средних τ.

### 4.3 Checkpoint cadence

Worker push-ит `(c_t, τ=0)` в gateway раз в K токенов или N секунд. **τ_emit = 0** потому что worker это authoritative state. Gateway хранит chain checkpoints.

При failover gateway передаёт новому воркеру `(c_chk, τ_estimated)` где τ_estimated растёт от 0 в момент checkpoint до ~τ_hi через 30 секунд (калиброванная функция).

### 4.4 Delta-checkpoint

Checkpoint chain хранит дельты. Поскольку capsule эволюционирует медленно (slots обновляются гейтированно), дельта между соседними checkpoint-ами компрессируется в 5-10× через quantization + run-length encoding.

Phase 0 это не реализуется. Phase 1+.

### 4.5 Tier как τ-axis

Один чекпоинт может обслуживаться разными tier-ами:
- "Edge tier" (0.5B): inference при τ_op = 1.0 (модель работает в шумном режиме, экономит compute).
- "Standard tier" (7B): τ_op = 0.1.
- "Oracle" (32B): τ_op = 0, multistep denoise.

Это *не* три разных модели, а одна model family, обученная на полном τ-диапазоне. Distillation реализуется самим диффузионным objective: модель учится мапить любую (c, τ) → c_0, что эквивалентно self-distillation between τ-режимами.

**Открытый вопрос:** одна архитектура с разными τ_op действительно ли даёт разные точки качества/compute? Это часть Phase 0 эксперимента.

---

## 5. Phase 0 эксперименты (8 недель)

### 5.1 Контрольные модели

| ID | Описание | Что меряем |
|---|---|---|
| `B1` | Vanilla capsule transformer (no diffusion, no augmentation) | Baseline PPL и failover quality |
| `B2` | Vanilla + capsule corruption augmentation | Effective ceiling adversarial-only подхода |
| `DST` | Полная DST модель | Нужна ли диффузионная параметризация |
| `DST-noLA` | DST без hierarchical lookahead | Изоляция вклада LA |
| `DST-noFO` | DST без synthetic failover loss | Изоляция вклада failover loss |

### 5.2 Эксперименты

**E1. Clean PPL.** Все 5 моделей на TinyStories validation. Цель: DST не отстаёт от B1 более чем на 5% в clean режиме.

**E2. PPL под τ-инъекцией.** Случайно инжектим noise τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} в произвольный момент сессии, меряем PPL на следующих 30 токенах. Цель: DST показывает плавную деградацию; B1 ломается при τ ≥ 0.5; B2 деградирует резче, чем DST, на τ выше тренировочного диапазона.

**E3. Recovery time.** После τ-инъекции, через сколько токенов PPL возвращается в пределы 5% от baseline. Цель: DST recovers за ≤ 5 токенов; baselines ≥ 20.

**E4. Failover simulation.** Симулируем смену worker: текущий c_t отбрасывается, новый воркер стартует с `c_{t-Δ} + noise`, Δ ∈ {10, 50, 100}. Меряем quality (PPL и downstream task acc) первых 50 токенов после failover.

**E5. Lookahead acceptance vs τ.** Acceptance rate hierarchical lookahead на 4 уровнях τ. Цель: показать самокалибровку (chunk size падает с ростом τ).

**E6. Tier-as-τ.** На одной обученной DST модели прогоняем inference при τ_op ∈ {0, 0.1, 0.5, 1.0, 2.0}, меряем quality vs compute (включая multistep при τ_op=0). Цель: monotonic trade-off curve.

### 5.3 Decision gate (gate G0 → Phase 1)

Phase 1 (scale to 0.5B + WAN simulator) запускается, если:
- E1: DST PPL within 5% of B1 — **must pass**.
- E3: DST recovery ≤ 5 tokens at τ=1.0 — **must pass**.
- E4: DST quality at Δ=50 failover within 10% of clean — **must pass**.
- E2 + E5 + E6: at least 2 of 3 показывают monotonic / smooth degradation — **should pass**.

Если G0 не пройден, fallback на baseline arch с adversarial augmentation (B2-style) и фиксируем "DST не выдержал" в `../architecture/failed_architectures.md`.

---

## 6. Открытые вопросы и риски

### 6.1 Технические

1. **Манифольд capsule states может быть irregular.** Diffusion работает на гладких манифольдах. Если capsule manifold имеет дискретную структуру (что возможно из-за gate-based update), CT может не сходиться. **Mitigation:** добавить spectral regularizer на capsule update; перейти на flow matching, если CT нестабилен.

2. **Bootstrap problem.** `c_0` нужен для тренировки denoiser, но `c_0` сам — выход модели. **Mitigation:** strict no_grad в teacher rollout (§3.4); inicialialize только c_0=0 на первой итерации.

3. **τ calibration.** Wall-clock сигналы сильно зашумлены, real-world τ распределение может отличаться от training. **Mitigation:** robust mixture of noise types в training (Gaussian + structured + masking); tune τ_observed на real WAN traces в Phase 1.

4. **Compute overhead.** Denoise + state update + LM = 3 forward stages вместо 1. Practical inference ≥ 1.5× стандартного капсульного transformer. **Mitigation:** weight sharing 8/12 слоёв; fused kernels во вторую очередь.

5. **Discrete tokens vs continuous capsule.** Diffusion на капсуле — continuous; sampling токенов — discrete. Эта декомпозиция работает в Stable Diffusion (latent diff + decoder), но capsule не декодирует напрямую в токены, а влияет через attention. Нужно убедиться, что градиент через discrete sampling корректно работает с continuous diffusion loss. **Status:** должно работать (LM_head — это просто softmax на детерминированной функции c_0_hat). Но проверить эмпирически.

### 6.2 Концептуальные

6. **Является ли τ действительно унифицирующей осью?** Сейчас τ объединяет: jitter, corruption, failover, tier. Возможно, эти явления требуют разных условий (например, корреляции в шуме при corruption ≠ независимый Gaussian). **Mitigation:** в этап C тренировки добавить смесь noise distributions (independent, structured, masking), conditioning не только на τ, но и на noise_type embedding.

7. **τ vs uncertainty.** Иуже есть `Uncertainty_head` для self-confidence. τ — это noise level state. Они разные понятия (τ — что я знаю про state, uncertainty — что я знаю про prediction). Должно работать ортогонально, но possible interference.

8. **Worker compute heterogeneity.** Если разные воркеры запускают модель с разным τ_op, gateway должен знать какой τ_op у воркера и compensate. Это runtime орchestration вопрос, не model вопрос.

### 6.3 Риски, ради которых я бы остановил исследование

- E3 не проходит на 0.1B (recovery > 10 tokens на τ=1.0): значит manifold assumption неверна для capsule.
- L_CT не сходится после curriculum stage C: значит CT-from-scratch несовместим с recurrent state. Откатываемся на flow matching или vanilla denoising autoencoder.
- E6 не показывает trade-off (qualité plateau across τ_op): значит "tier-as-τ" не работает, нужны явно разные модели для разных tier.

---

## 7. Что строить первым (8 недель Phase 0)

| Неделя | Цель | Артефакт |
|---|---|---|
| 1 | TinyStories baseline B1 (vanilla capsule) | repo, training loop, B1 ckpt, val PPL |
| 2 | + capsule corruption augmentation B2 | B2 ckpt, E2 предварительно |
| 3 | DST skeleton: AdaLN(τ), denoise stage, CT loss | Sanity: tiny model сходится на toy task |
| 4 | DST full training stage A+B | Mid-training validation |
| 5 | DST stage C (full τ диапазон) | Финальный ckpt |
| 6 | Эксперименты E1, E2, E3 | Recovery curves |
| 7 | Эксперименты E4, E5, E6 | Failover + tier-as-τ tables |
| 8 | Decision gate G0, ретроспектива, draft Phase 1 plan | Решение: scale / pivot / kill |

### 7.1 Минимальный stack

- PyTorch + Lightning или аналог (vanilla loop тоже ок).
- xformers / FlashAttention для скорости.
- HuggingFace tokenizers (BPE 32K).
- Datasets: TinyStories (HF), RedPajama-v2 sample.
- Compute: 8× A100-80GB на 5 дней — реалистично через cloud rent.

### 7.2 Что реализуется первым в коде

1. `Capsule` (just a tensor wrapper with shape contract).
2. `DSTBlock` (transformer block + AdaLN(τ)).
3. `DSTModel.forward(c_obs, τ, window) → c_0_hat, U_out, lm_logits, lookahead_logits, uncertainty`.
4. `EDMNoiseSchedule` (sample τ, alpha/sigma, preconditioning).
5. `train_step` с full L_total.
6. `synthetic_failover_batch` сэмплер.
7. Eval suite: E1-E6 как отдельные scripts.

Каждый из этих компонентов изолированно тестируется на toy task (например, "capsule учится запоминать число из начала последовательности" — память-needle на 96 токенах).

---

## 8. Связанные работы для review

Перед стартом стоит прочитать:

- **Karras et al, EDM (2022)** — preconditioning, noise schedule.
- **Song et al, Consistency Models (2023)** — CT objective, EMA target.
- **Peebles & Xie, DiT (2023)** — AdaLN conditioning в transformer.
- **Liu et al, Rectified Flow (2023)** — fallback, если CT нестабилен.
- **Gu et al, Mamba (2023)** — recurrent state в SSM, контекст для capsule design.
- **Bulatov et al, RMT (2022)** — recurrent memory transformer, ближайший родственник capsule arch.
- **Fan et al, EAGLE-2 (2024)** — иерархический lookahead.
- **Guo et al, "Memorizing Transformers" (2022)** — alternative для long-term retrieval.

---

## 9. Что **не** делается в Phase 0

Чтобы scope не расплылся:

- Никакой quantization-aware training (Phase 1+).
- Никакого WAN simulator (Phase 1).
- Никакого multi-session batching (Phase 2).
- Никаких real workers / gateway / failover infra (Phase 1+).
- Никакого hierarchical lookahead head_2/head_3 если head_1 не работает.
- Никакой VAE-latent capsule (только если CT на raw слотах не сходится).

Phase 0 — чисто алгоритмический gate: работает ли диффузия на capsule state.

---

**Owners:** TBD.
**Last updated:** 2026-04-28.
