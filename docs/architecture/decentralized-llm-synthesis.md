---
title: "Decentralized LLM — Синтез исследования"
date: 2026-04-29
author: TBD
version: v1.0
status: living-document
stream: wan
tags: [synthesis, architecture, DST, capsule, roadmap, single-source-of-truth, wan]
---

# Decentralized LLM — Синтез исследования

**Назначение документа.** Сводка всех обсуждений по архитектуре децентрализованного LLM-инференса с capsule state и Diffusion-State Transformer. Этот документ — single source of truth и навигатор по другим файлам в проекте. Все детальные спеки лежат в отдельных файлах, здесь — концептуальная картина целиком.

**Last updated:** 2026-04-29.

---

## 0. Навигация по документам

| Файл | Назначение |
|---|---|
| `columnar-decentralized-llm.md` | Исходная архитектура: Summary-State Transformer + Session State Capsule. Базовый дизайн, 10 разделов. |
| `failed_architectures.md` | Кладбище подходов. 9 паттернов, которые не работают на WAN, с математикой почему. |
| `../research/dst-research-spec.md` | Полная research-спека Diffusion-State Transformer. Math, потери, эксперименты Phase 0. |
| `../research/dst-engineering.md` | Операционный слой DST: deployment, inference path, trust, observability. |
| `decentralized-llm-synthesis.md` | (этот файл) Синтез всего обсуждённого. |

---

## 1. Контекст и цели

**Что строим.** LLM-инференс через P2P/WAN сеть consumer GPU для клиентов без VRAM. Целевая скорость ~15 tok/s на пользователя.

**Trust boundary.**
- **Managed control plane** (наш контроль): Gateway, Router, State Store, Oracle Pool, Model Registry, Telemetry.
- **Untrusted P2P workers**: consumer GPU (3060, 4090, и т.п.) для compute. Не источник доверия.

**Главный design constraint.** Авторегрессионная генерация batch=1 + WAN с RTT 50-150ms делают **синхронные межузловые операции на каждый токен невозможными**. Любая архитектура, требующая cross-worker sync per token, обречена (см. §5 и `failed_architectures.md`).

**Целевая математика (из исходного документа):**
```
RTT до worker:          80-120 ms
worker lookahead chunk: 40-80 ms
gateway overhead:        5-20 ms
total chunk:           140-220 ms

3 accepted tokens / 200 ms ≈ 15 tok/s
```

KPI: **≥ 3 accepted tokens per remote chunk** при RTT ~100 ms.

---

## 2. Исходная архитектура (Summary-State Transformer)

Полное описание в `columnar-decentralized-llm.md`. Краткая выжимка:

**Идея.** Вместо raw KV-cache (который растёт линейно с контекстом и его дорого мигрировать) — модель с компактным state ("Session State Capsule"), специально обученная под этот state contract.

**Capsule содержит:**
- Memory slots (фиксированное число).
- Recent sliding-window tokens.
- Position / routing metadata.
- Optional integrity metadata.

**Целевые размеры:**

| Model | Sliding window | Memory slots | Capsule | Checkpoint cadence |
|---|---:|---:|---:|---:|
| 0.5B | 128-256 | 16-32 | 50-200 KB | 20-50 tokens |
| 1.5B | 256-512 | 32-64 | 200-800 KB | 20-50 tokens |
| 7B | 512-1024 | 64-128 | 1-5 MB | 50-100 tokens |

**Жизненный цикл сессии:**
- Worker hot-stateful: capsule живёт локально на воркере во время сессии.
- Checkpoints в State Store асинхронно (не per-token).
- Failover: новый worker получает последний capsule, продолжает.
- Cold fallback: при stale/lost capsule — full re-prefill (UX freeze 1-3 сек).

**Lookahead heads** живут на воркере (Medusa/EAGLE-стиль). Один RTT возвращает chunk токенов, не один.

**Качество масштабируется** только через tier escalation, managed oracle, best-of-N, specialist critics. Replica count сам по себе **не делает одну сессию умнее**.

---

## 3. Критика исходной архитектуры

Что в исходном документе хорошо:
- Capsule как checkpoint, а не per-token payload.
- Lookahead на воркере (gateway без GPU).
- Честное признание: больше реплик = больше throughput, не качество.

Что слабо или умалчивается:

1. **Capsule contract — это research, не engineering.** Документ это признаёт, но не оценивает риск провала. Memory slots на 64-128 для 7B — компромисс уровня RWKV/Mamba/RetNet, и их quality на long-context reasoning до dense baseline не дотягивает. Нужен явный go/no-go gate после 0.5B прототипа.

2. **Batch=1 на 4090 — недогруз GPU.** Документ не упоминает multi-session batching. Без него ROI на consumer GPU плохой.

3. **Trust в Workers не раскрыт.** Сказано "key-locking metadata", но протокола нет. Без verifier sampling и cryptographic commits приватность держится на честном слове.

4. **Cold start длинного контекста.** Если у пользователя 32K промпт, prefill на удалённом воркере по WAN — десятки секунд. Не описано, кто и где prefill делает.

5. **Sticky sessions vs load balancing.** Hot-stateful = пользователь привязан к воркеру. Если воркер занят — куда новую сессию? Fragmentation воркеров и hot-spot проблемы не разобраны.

6. **Lookahead acceptance оптимистичен.** "≥3 accepted/chunk" — для лёгких текстов. На коде/математике/длинном контексте acceptance падает до 1.5-2. KPI должен быть p50 *и* p10.

7. **Quantization × distillation interaction.** 4/8-bit serving кастомной Summary-State модели требует QAT с самого начала. Не упомянуто.

---

## 4. Фундаментальный вопрос: можно ли разделить модель?

Главный архитектурный вопрос: на каждом воркере полная копия модели — это много VRAM. Хочется разделить веса между воркерами. **Можно ли?**

### 4.1 Короткий ответ

**Нет.** Не через WAN. Любая форма parameter sharding (Tensor Parallel, Pipeline Parallel, MoE expert sharding) для batch=1 авторегрессии требует per-token sync через сеть, что упирается в RTT и даёт 1-3 tok/s вместо 15.

Это не "потеря 10-20%", а **5-50× медленнее**. Документировано в `failed_architectures.md` пункты 3, 4, 5, 7.

### 4.2 Что **можно** делать вместо

| Подход | Где работает | Цена |
|---|---|---|
| **Локальный sharding на одном воркере** | Q4/Q8 quantization, NVMe weight streaming, CPU offload FFN, layer offload + prefetch | 0-30% speed, 4-10× memory savings |
| **Micro-pods** (2-4 воркера на LAN) | TP внутри пода с RTT < 5ms; снаружи pod = один логический воркер | Discovery сложен, ~5-10% сети сможет образовать pod |
| **Гетерогенная репликация tier** | Разные воркеры держат разные размеры одной model family (1.5B / 7B / 14B) | Полная репликация внутри tier; нет sharding весов |
| **Функциональное разделение** | Specialists (code/math/safety) — отдельные модели; orchestration через gateway, async | Дополнительные training pipelines |
| **Retrieval вместо memory sharding** | Long-term memory во внешнем vector store (managed), capsule остаётся локальной | Дополнительная инфра |

### 4.3 Принятый design choice

**Полная репликация внутри tier** + **гетерогенные tier по сети** + **локальный sharding (Q4 + offload) для влезания в VRAM**.

Это значит: на 4090 крутится полная копия 7B Q4. На 3060 — полная копия 1.5B Q8. Никакого пересечения forward pass через сеть.

---

## 5. Failure modes (расширенный список)

Из `failed_architectures.md`, 9 паттернов, которые не работают:

1. **Naive frozen ensemble** — усреднение независимых колонок. Модели делают одинаковые ошибки, усреднение даёт высокую энтропию = мусор.

2. **Per-token full logits voting** — каждый токен передаёт полные logits всех воркеров. 800 Mbps на мастера. Bandwidth death.

3. **Micro-pipeline parallelism через интернет** — слои распределены по воркерам. Авторегрессия = feedback loop, RTT × stages per token. Pipeline не работает в цикле с обратной связью.

4. **Time-Bounded Columnar Pipeline (TBCP)** — глубокая сетка с тайм-аутами. Stateless воркеры в слоистом pipeline невозможны (KV-кэш). Bufferbloat на consumer Upload.

5. **TP-Surgical (Tensor Parallel) через Wi-Fi** — 6 round-trips на токен при RTT 50-100ms = 1.5-3 tok/s. Работает только в LAN/colocated.

6. **Stateful KV-cache thrashing** — слабые ноды бесконечно догоняют сеть, никогда не приносят пользы.

7. **Idealized speculative decoding** — реальный acceptance 2-4 токена, не 10. Реальная скорость ~10-15 tok/s, не 50.

8. **WAN MoE с экспертами на разных воркерах** — per-token routing, RTT до выбранного эксперта. Для 24 слоёв MoE → 4.8 сек/токен при RTT 100ms. Работает только в датацентре.

9. **Шардинг memory slots / capsule между воркерами** — read-after-write per step через сеть. 5-10 tok/s в лучшем случае. Async-чтение ломает recurrent contract.

10. **Async pipeline с rollback ("confidence-bounded continuation")** — A предсказывает что вернёт B, при mismatch rollback. Acceptance ~30-50%, throughput хуже синхронного pipeline. По сути spec decoding в обёртке pipeline.

**Итог.** Все попытки заставить P2P-сеть работать синхронно на каждом токене — провал из-за задержек. Параметрический sharding фундаментально несовместим с WAN при batch=1. Децентрализация требует:
- Репликации, а не sharding весов.
- Асинхронных протоколов на уровне ролей (target/draft/oracle/critic), а не слоёв.
- Micro-pods для случаев, когда sharding всё-таки нужен (внутри pod RTT < 5ms).
- Локальных решений (quantization, NVMe offload) для проблемы "модель не лезет в VRAM".

---

## 6. Улучшения: стабильность

1. **Delta-checkpoint** — передавать diff capsule, не весь. Экономит ~80% bandwidth.

2. **Shadow-replica для премиум-сессий** — второй воркер держит warm state (capsule sync), но не генерирует. Failover мгновенный. Цена: +50% compute на сессию.

3. **Verifier sampling** — Gateway re-runs 1-2% токенов на trusted oracle, сравнивает logits. Защита от malicious workers.

4. **Cryptographic capsule chain** — каждый checkpoint несёт hash предыдущего. Защита от форка conversation.

5. **Tail-latency aware migration** — миграция при p95 RTT > threshold или дрейфе acceptance rate, до фактического fail.

6. **Capsule schema versioning** — любое изменение в schema = major version. Mismatched воркеры не получают сессии.

7. **Predictive churn signal** — pre-warm нового воркера до того как старый умер (GPU temp, network jitter, last-seen heartbeat).

---

## 7. Улучшения: скейлинг

1. **Multi-session batching на воркере** — 1 × 7B на 4090 → 4-8 concurrent sessions через continuous batching. Critically важно, без этого экономика P2P провальная.

2. **Prompt-prefix dedup** — если 100 сессий стартуют с одного system prompt, capsule этого префикса считается один раз и шарится.

3. **Tier autoscaler на gateway** — маршрутизация на дешёвый tier по difficulty estimator. 0.5B берёт ~60% токенов; 7B — ~30%; 14B oracle — ~10%.

4. **Heterogeneity-aware scheduler** — GPU фингерпринтится. Сложные сессии не идут на 3060.

5. **Speculative capsule prefetch** — перед миграцией capsule push'ится в edge cache рядом с потенциальными следующими воркерами.

---

## 8. Улучшения: качество

1. **Process-level oracle critique** — oracle верифицирует CoT-цепочки на промежуточных steps, не только финальный ответ.

2. **Best-of-N в spare capacity** — при utilization < 50% параллельные drafts с разным seed; oracle reranks. Бесплатное улучшение качества.

3. **External memory вместо capsule expansion** — long-term context в vector store (managed); capsule имеет указатели в этот store. Честнее, чем заставлять recurrent state помнить всё.

4. **Mixture of critics, не mixture of experts** — specialist воркеры (code/math/safety) делают post-hoc re-ranking, не участвуют в forward pass. Не нарушает state contract.

5. **Test-time memory adaptation** — лёгкое online-обновление memory update rule (LoRA на ~MB) per-session. Adapter тоже в capsule, не ломает portability.

6. **Self-uncertainty gating** — модель сама выдаёт uncertainty score, gateway escalate spans с высокой uncertainty на oracle. Token-level escalation, не request-level.

7. **Hierarchical lookahead** — heads предсказывают POS/struct class → token class → token. Acceptance каскад поднимает avg accepted tokens/chunk с 3 до 5-6.

---

## 9. Три research bet

Текущая capsule-arch требует обучения модели под state contract — это high-risk research. Альтернатива retrofit-augmentation: **архитектура, в которой failure modes — это inductive bias, а не корректирующая настройка**.

Три ставки, рассмотренные на брейншторме. (1) — главная, (2) и (3) — hedge.

### 9.1 Bet 1: Diffusion-State Transformer (DST) ⭐ главная ставка

Capsule эволюционирует как denoising diffusion. Все failure modes (jitter, corruption, failover, tier) объединены одной осью **τ ∈ [0, T_max]** — noise level.

| Failure mode | Как DST это решает |
|---|---|
| Worker failover | Stale capsule = high-τ. Воркер денойзит на лету. |
| Capsule corruption | Литерально training distribution. |
| Checkpoint jitter | τ растёт со временем без checkpoint. Модель самокалибруется. |
| Limited memory slots | τ-conditioned bottleneck. |
| Lookahead acceptance | Lookahead head условится на τ, conservative при noisy state. |
| Tier/oracle distillation | Tier — это разные точки на τ-оси. Один model family. |

Полная спека: `../research/dst-research-spec.md`. Краткое содержание в §10 этого документа.

### 9.2 Bet 2: Neural-ODE Continuous-Time Capsule (NOC) — hedge

Capsule — латентная переменная z(t), эволюционирующая по learned ODE: `dz/dt = f_θ(z, x_t, t)`.

- Failover: продолжаем интегрировать с последнего checkpoint.
- Checkpoint jitter: нативно (Δt — параметр solver).
- Tier: ODE solver step-size policy.

Связи: Mamba — это discrete-time приближение к этой идее. NOC — менее радикально, чем DST.

### 9.3 Bet 3: Holographic Memory + Retrieval (HoloCap) — hedge

Memory slots — superposition фактов через Holographic Reduced Representations (Plate, '95). Каждый слот хранит ~100 вещей, извлечение через correlation.

- Failover с потерей K% слотов: graceful degradation благодаря HRR redundancy.
- Limited slots: HRR — почти теоретико-информационный оптимум.
- Дополнительно: то что не лезет — в managed retrieval store.

Менее sexy чем DST, но более робастно к research uncertainty.

---

## 10. DST в глубину

Полная спека в `../research/dst-research-spec.md`. Здесь — концептуальный каркас.

### 10.1 Формализация

Forward (noising): `c_τ = α(τ) · c_0 + σ(τ) · ε`, ε ~ N(0, I). EDM-параметризация (Karras).

Reverse: denoiser `D_θ(c_τ, τ, w)` обучается через Consistency Training (Song et al). На inference — 1 step (или 2 при высоком τ).

τ маппится на runtime сигналы:
```
τ_observed = clip(
    a · seconds_since_checkpoint
  + b · transmission_compression
  + c · packet_loss_indicator
  + d · failover_recency
  , 0, T_max
)
```

### 10.2 Архитектура

Backbone — pre-norm transformer на conjoined sequence `[slots; window]`. AdaLN(τ) — sinusoidal embedding(τ) → (γ, β) для каждого LayerNorm.

Phase 0 параметры: 32 slots × 256 dim, sliding window 64 tokens, 12 layers, ~120M params.

Головы:
- D_θ — denoiser
- U_θ — state update
- LM_head — next token
- Lookahead_k (k=1..3) — спекулятивные head
- Struct_head — coarse token class
- Uncertainty_head — self-uncertainty

### 10.3 Тренировка

7 потерь:
- L_LM — стандартный CE
- L_CT — consistency training (denoising)
- L_state_consis — denoise + update коммутируют с clean update
- L_lookahead — спекулятивные head
- L_struct — hierarchical lookahead
- L_failover — synthetic failover trajectories
- L_uncertainty — self-uncertainty regression

Curriculum: A (LM only, τ=0) → B (+ state consis + lookahead, τ ∈ [0,1]) → C (всё, τ ∈ [0, T_max=80]).

### 10.4 Phase 0 эксперименты (8 недель)

5 моделей сравниваются: vanilla baseline (B1), + adversarial augmentation (B2), полная DST, DST без lookahead, DST без failover loss.

6 экспериментов: clean PPL, PPL под τ-injection, recovery time, failover quality, lookahead acceptance vs τ, tier-as-τ trade-off curve.

**Decision gate G0** для перехода в Phase 1:
- DST PPL within 5% of B1 — must pass
- Recovery ≤ 5 tokens at τ=1.0 — must pass
- Failover quality at Δ=50 within 10% of clean — must pass

Если G0 не пройден — fallback на adversarial augmentation, фиксируем "DST не выдержал" в `failed_architectures.md`.

### 10.5 Открытые риски DST

- Manifold капсул может быть irregular, CT может не сходиться.
- Bootstrap problem (c_0 — выход модели).
- τ calibration на real-world traces.
- Compute overhead денойзинга (1.5× стандартного inference).
- Quantization × τ — Q4 может не сохранить denoising качество на high τ.

---

## 11. DST inference & deployment

Полная спека в `../research/dst-engineering.md`. Здесь — основные инварианты.

### 11.1 Топология

```
Client (zero-VRAM)
    │
    ▼
Managed Control Plane:
  Gateway, Router, State Store,
  Difficulty Estimator, Oracle Pool,
  Telemetry, Model Registry
    │
    │ (signed commands, capsules over QUIC)
    ▼
P2P Workers:
  Worker A (4090, DST 7B Q4, τ_op=0.1)
  Worker B (3060, DST 1.5B Q8, τ_op=0.5)
  Worker N (4090, specialist code 1.5B Q8)
```

### 11.2 Инвариант: воркеры между собой не общаются

**Никогда.** Только через Gateway. Это держит всю сетевую модель — иначе попадаем в WAN sync trap из §5.

### 11.3 Bundle модели

```
dst-7b-r2.4/
├── manifest.json              ← версии, hashes, compatibility
├── weights.safetensors        ← Q4 GPTQ или fp16
├── tokenizer.json
├── tau_calibration.json       ← runtime → τ_observed
├── capsule_schema.json        ← N, D, slot layout
├── lookahead_config.json
├── consistency_steps.json
└── runtime_contract.json      ← min/recommended VRAM
```

Capsule schema versioning — критично. Любое изменение в N/D/slot semantics — major bump. Workers с несовместимой schema получают 0% трафика.

### 11.4 VRAM бюджет (RTX 4090, 24GB)

| Компонент | VRAM (Q4) |
|---|---:|
| DST 7B веса | 4.0 GB |
| KV-cache (64 tok × 6 sessions × 32 layers) | ~3 GB |
| Capsule × 6 sessions | < 1 MB |
| Activations peak | ~2.5 GB |
| Lookahead trees | ~500 MB |
| CUDA workspace | ~1 GB |
| **Итого** | **~11 GB** |

Запас 10 GB на разворачивание Q4 в compute kernels + warm next bundle для rollout.

### 11.5 Per-step протокол

QUIC long-lived connection. Один step = один RTT возвращает chunk + lookahead + uncertainty + capsule delta:

```
StepRequest:
  session_id, user_input_tokens, last_accepted_seq, control_flags

StepResponse:
  accepted_tokens, structural_commits, self_uncertainty,
  capsule_seq, capsule_delta?, tau_observed?, worker_health
```

### 11.6 Когда денойзить (политика)

```
if τ_internal < 0.1:   skip_denoise        # +25% throughput на стационарных
elif τ_internal < 1.5: denoise_1step       # обычный режим
else:                  denoise_2step       # после failover / длительного offline
```

### 11.7 Multi-session batching

Continuous batching (vLLM-style). Capsule всех sessions упакован в `batch[B, N, D]`, τ — per-session scalar broadcast в AdaLN. KV-cache — paged.

Без батчинга 4090 утилизирована на 15-20%. С батчингом — 60-80%. **Критично для экономики.**

### 11.8 Cold start vs warm continuation vs cold fallback

- **Cold start**: prefill промпта, push первый checkpoint.
- **Warm continuation**: денойзим capsule из checkpoint, replay missing tokens.
- **Cold fallback**: capsule lost/incompatible → full re-prefill всего диалога. Freeze 1-3 сек, видимый клиенту.

---

## 12. Гетерогенная репликация tier-ов и роли воркеров

### 12.1 Кто что держит

| Tier | Размер | Носитель | % сети |
|---|---:|---|---:|
| Edge | 1.5B Q8 | 12GB GPU (3060/4060) | 60-70% |
| Standard | 7B Q4 | 24GB GPU (4090/3090) | 25-30% |
| Oracle | 14-32B | managed datacenter | < 5%, наш контроль |
| Specialists | 1-3B | mix | 5-10% |

В Phase 0/1 — это **отдельно обученные** модели одной DST-семьи (общая capsule schema). В Phase 2+ — option на nested training (см. §13).

### 12.2 Веса между воркерами

Не пересекаются никогда. Каждый воркер держит **полные** веса своей модели и запускает forward pass у себя локально.

### 12.3 Cross-worker координация (только асинхронная)

Внутри одного токена активен ровно один воркер (primary generator). Через жизненный цикл сессии могут участвовать несколько, координируемых Gateway:

| Паттерн | Когда | Как |
|---|---|---|
| Primary | базовая генерация | один воркер, full forward, hot capsule |
| Oracle escalation | uncertainty spike, hard span | gateway параллельно дёргает oracle, его вывод переписывает следующий chunk (не текущий, чтобы не было видимого rollback) |
| Specialist critique | post-hoc на code/math/safety span | специалист пере-валидирует уже принятые токены, при disagreement primary regenerates |
| Best-of-N | utilization < 50% | 2-3 воркера независимо генерируют, gateway picks best |
| Verifier sampling | 1-2% случайных токенов | re-run на oracle, сравнение logits, blacklist при расхождении |
| Failover | primary упал | gateway достаёт capsule, передаёт другому воркеру того же tier |

Все паттерны **асинхронны** относительно per-token цикла.

### 12.4 Fundamental design choice (резюме)

**Реплицировать полные модели разного размера** + **Gateway-orchestrated асинхронная композиция**.

НЕ: разделять одну модель + синхронная композиция.

---

## 13. MatFormer × DST: nested training (Phase 2+)

### 13.1 Идея на пальцах

Матрёшка. Большая модель, в которой меньшие — её собственные подмножества параметров. Каждая "обрезанная" версия — законченная модель сама по себе.

В FFN слое: вместо обычного `вход(4096) → hidden(16384) → выход(4096)` обучаем так, что **первые 1024 нейрона hidden** работают как самостоятельная модель (≈0.5B), **первые 4096** — как 1.5B, **все 16384** — как 7B.

### 13.2 Как тренируется

На каждом батче случайно выбираем размер:
- Батч 1: forward через все 16384 → 7B
- Батч 2: forward через первые 4096 → 1.5B
- Батч 3: forward через первые 1024 → 0.5B
- Backward только через выбранный размер

После долгой тренировки все размеры — валидные модели в одном наборе весов.

### 13.3 Что это даёт DST

1. **Один бандл вместо трёх** — 0.5B/1.5B/7B = срезы одних весов. Один тренировочный процесс, один файл, один деплой.

2. **Воркер обслуживает любой tier динамически** — 4090 загрузил один файл; пришёл лёгкий запрос → forward через 25% нейронов в 4× быстрее; пришёл тяжёлый → full forward. Без перезагрузки.

3. **Tier-as-τ становится буквальным** — размер слайса физически соответствует τ_op. Меньше нейронов = меньше denoising power = выше эффективное τ_op.

4. **Tier-switch внутри сессии** — воркер может на лету переключиться между slice-ами, capsule общая (одни веса).

### 13.4 Цена

- Каждый slice ~2-5% хуже dedicated модели того же размера (MatFormer paper).
- Тренировка +30-50% compute.
- Архитектурное ограничение: глубина одна для всех slice; режется только ширина.
- Specialists остаются отдельно (другой домен данных).

### 13.5 Почему не сразу

Совмещение MatFormer (multi-size joint training) с DST (diffusion на capsule) — две research challenges одновременно. Делаем поэтапно:
- Phase 0: один размер, проверяем DST.
- Phase 1: 0.5B + 7B как отдельные тренировки, общая capsule schema.
- Phase 2: research project на MatFormer × DST. +6 месяцев.

---

## 14. Trust & integrity

### 14.1 Worker attestation

При registration воркер предоставляет:
- Подписанный сертификат (наш CA).
- Hardware fingerprint (GPU UUID, MAC).
- Software hash (binary + bundle).
- Optional TPM/SGX attestation.

Quarantine pool для подозрительных, blacklist при накоплении negative repute.

### 14.2 Capsule integrity

Каждый checkpoint:
```
{
  capsule_bytes,
  prev_checkpoint_hash: sha256(...),
  worker_signature: sign(capsule_bytes || prev_hash, worker_key),
  sequence_no
}
```

Chain commit. Rewrite старой capsule ломает hash chain → State Store отвергает. Защита от malicious rewind.

### 14.3 Probabilistic verifier sampling

1-2% токенов re-run на oracle. KL(p_worker || p_oracle) > threshold → миграция, negative repute.

### 14.4 Что НЕ обеспечивается

- **Privacy от worker.** Worker видит plaintext. Без homomorphic encryption (нерабочий compute budget) не лечится. Mitigation: классификация запросов — sensitive не идёт на untrusted P2P.
- **Deterministic guarantee qualité.** Только probabilistic через sampling.

---

## 15. Roadmap

### Phase 0 (8 недель) — DST research gate

Цель: доказать что diffusion на capsule state работает.

- Неделя 1-2: B1 baseline (vanilla capsule) на TinyStories.
- Неделя 3-5: DST training (curriculum A→B→C).
- Неделя 6-7: эксперименты E1-E6.
- Неделя 8: G0 decision.

Compute: ~8× A100 × 5 дней.

**G0 gate:** DST recovers ≤ 5 токенов от τ=1.0, failover quality within 10%, PPL within 5% от baseline.

Если не пройден → fallback на adversarial augmentation на vanilla capsule arch.

### Phase 1 (3-4 месяца) — MVP

Цель: end-to-end на 0.5B + 50 workers в реальной сети.

- DST 0.5B и 7B обучены отдельно, общая capsule schema.
- Single-tier routing (только standard).
- Cert-based attestation, без TPM.
- Capsule chain hash, без KL verifier sampling.
- TLS+gRPC, без QUIC.
- Postgres+S3 для capsule storage.
- Multi-session batching на воркере.
- Cold fallback работает.

### Phase 2 (~6 месяцев) — Production hardening

- Multi-tier с escalation.
- QUIC.
- Best-of-N + verifier sampling.
- Multi-region.
- Specialist critics (code/math/safety).
- Capsule schema v2 с запасом.
- **Research:** MatFormer × DST joint training.

### Phase 3 (~год) — Scale

- 1000+ workers.
- Capsule schema migration tooling (live, не cold drain).
- Federated training updates (если делаем).
- Token-level oracle escalation на base path.

---

## 16. Открытые вопросы / решения, которые ещё надо принять

### Архитектурные

1. **MatFormer × DST совместимы?** Phase 2 research project. Если нет — остаёмся на отдельных моделях.
2. **Capsule schema финальные параметры (N, D)?** Phase 1 commit. Ошибка = шлейф годами.
3. **Specialist capsule-aware или text-only?** Phase 1 = text-only, Phase 2 = ?

### Инженерные

4. **NAT traversal протокол.** Outbound-only QUIC tunnel, но детали (keep-alive, reconnect, multiplexing) — открыто.
5. **Multi-session batching engine.** Custom или адаптация vLLM?
6. **AdaLN(τ) в batched режиме** — нужен custom CUDA kernel или сегрегация по τ-bucket?
7. **Quantization × denoising качество** — Q4 переживёт или нужно fp8 на denoiser?

### Operational

8. **Worker churn модель.** Consumer GPU = нерегулярная доступность. Routing-стратегия для long sessions vs bursty запросов.
9. **Compensation модель воркеров.** Кто и как платит pro-sumer за compute. Не архитектурный, но влияет на retention.
10. **Privacy classification протокол.** Как классифицируем sensitive vs non-sensitive запросы.

### Research

11. **Что если DST не пройдёт G0.** Fallback на adversarial augmentation — но насколько это конкурентно? Возможно нужен hybrid с NOC (Bet 2).
12. **Token-level uncertainty calibration.** Self-uncertainty head — как калибровать с реальной KL до oracle?
13. **Capsule corruption distribution в реальном WAN.** Synthetic noise на тренировке vs реальные packet loss / quantization errors.

---

## 17. Глоссарий

| Термин | Определение |
|---|---|
| **Capsule** | Компактное переносимое state сессии: memory slots + sliding window + metadata. |
| **Session State Capsule** | Полное название capsule в исходной архитектуре. |
| **Memory slots** | Фиксированное число векторов памяти, обновляемых recurrent/attention update. |
| **Sliding window** | Recent W токенов, видимых полным attention. |
| **Lookahead heads** | Спекулятивные heads (Medusa/EAGLE-стиль) на воркере, генерируют chunk токенов за RTT. |
| **Tier** | Уровень модели (edge / standard / oracle / specialist). |
| **DST** | Diffusion-State Transformer. Главная research-ставка. |
| **τ (tau)** | Noise level capsule. Continuous axis [0, T_max=80]. |
| **τ_op** | Operational τ — на каком noise level worker запускает inference. |
| **AdaLN** | Adaptive LayerNorm, condition on τ via learned (γ, β) shift/scale. |
| **CT** | Consistency Training (Song et al, 2023). Способ обучить 1-step denoiser. |
| **EDM** | Elucidating Diffusion Models (Karras et al, 2022). Preconditioning для diffusion. |
| **MatFormer** | Matryoshka Transformer (Google, 2023). Nested training, slice-as-model. |
| **Gateway** | Managed proxy между client и worker. Orchestrator. |
| **State Store** | Managed storage для capsule chain. Hot tier + cold S3. |
| **Oracle** | Managed большая модель для escalation на hard tokens/spans. |
| **Specialist** | Domain-specific small model (code/math/safety) для post-hoc critique. |
| **Difficulty Estimator** | Лёгкая модель на gateway, оценивает сложность запроса для tier routing. |
| **Verifier sampling** | Probabilistic re-run K% токенов на oracle для trust. |
| **Cold fallback** | При lost/incompatible capsule — full re-prefill всего диалога. UX freeze. |
| **Warm continuation** | Failover через capsule из State Store, denoise + replay missing tokens. |
| **Micro-pod** | 2-4 воркера на LAN с RTT < 5ms, выглядят как один логический воркер с TP внутри. |
| **WAN** | Wide Area Network — публичный интернет с RTT 50-150ms. |
| **TP** | Tensor Parallelism — sharding одного forward pass по dimensions. |
| **PP** | Pipeline Parallelism — sharding по слоям. |
| **MoE** | Mixture of Experts — sparse routing к подмножеству параметров. |
| **HRR** | Holographic Reduced Representations — superposition фактов в одном слоте. |
| **MQA / GQA** | Multi/Grouped Query Attention — варианты attention с меньшим KV cost. |

---

## 18. Резюме одной страницей

**Что строим:** P2P LLM-инференс на consumer GPU, ~15 tok/s, для клиентов без VRAM.

**Главный design choice:** полная репликация моделей внутри tier + гетерогенные tier по сети + асинхронная Gateway-orchestrated композиция. **Никакого parameter sharding через WAN.**

**Главная research-ставка:** Diffusion-State Transformer. Capsule эволюционирует через denoising; failover/jitter/corruption/tier — точки на одной оси τ. Research gate G0 после Phase 0 (8 недель).

**Если DST не сработает:** fallback на vanilla capsule arch с adversarial augmentation. Менее элегантно, но работоспособно.

**Phase 2 amplifier:** MatFormer-style nested training. Один файл весов содержит все tier как срезы. Матрёшка.

**Главный enabler экономики:** multi-session batching на воркере (vLLM-style continuous batching), без него ROI на consumer GPU плохой.

**Главное trust ограничение:** privacy от worker не обеспечивается (homomorphic crypto не вписывается в compute budget). Sensitive traffic классифицируется и идёт только на managed tier.

**Главный архитектурный инвариант:** воркеры между собой никогда не общаются. Только через Gateway. Только асинхронно. Только между токенами/spans, никогда внутри одного токена.
