---
title: "DST Engineering: Deployment & Inference"
date: 2026-04-28
author: TBD
version: v1.0
status: research-draft
stream: wan
tags: [DST, engineering, deployment, inference, p2p, workers, wan]
---

# DST Engineering: Deployment & Inference

**Контекст.** Этот документ — инженерный слой поверх `dst-research-spec.md`. Он описывает, как DST-модель разворачивается на consumer GPU в P2P-сети и как обслуживает реальные сессии. Предполагается, что Phase 0 gate G0 пройден; модель существует, остаётся её эксплуатировать.

Связь с trust boundary из `../architecture/columnar-decentralized-llm.md`:
- **Managed** (наш control plane): Gateway, State Store, Oracle, Model Registry, Telemetry.
- **Untrusted** (P2P): Workers на consumer GPU.

---

## 1. Топология

```
                 ┌──────────────────────────────────────────────┐
                 │  MANAGED CONTROL PLANE (наша инфра)          │
                 │                                              │
   Client ───────►  Gateway ────► Router ────► State Store      │
   (zero-VRAM)     │   │             │             │            │
                  │   │             ▼             │            │
                   │   │         Difficulty       │            │
                   │   │         Estimator        │            │
                   │   │             │            │            │
                   │   │             ▼            │            │
                   │   └─────► Oracle Pool ◄──────┤            │
                   │       (managed 14B/32B)      │            │
                   │                              │            │
                   │       Telemetry / Health ◄───┘            │
                   └──────────────────────────────────────────────┘
                                 │  
                          (signed commands,                 
                           capsules over QUIC)              
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
            Worker A         Worker B         Worker N    ← P2P
            (RTX 4090)      (RTX 3060)       (RTX 4090)    consumer GPU
            DST 7B Q4       DST 1.5B Q8      DST 7B Q4
            τ_op = 0.1      τ_op = 0.5       τ_op = 0.1
```

Ключевые свойства:
- Только Gateway/Oracle/Store разговаривают друг с другом. Workers не общаются между собой.
- Workers подписаны нашим CA и идентифицируют себя сертификатом + attestation (см. §6).
- Все capsule transit идут через managed plane. Worker никогда не отправляет capsule напрямую другому worker.

---

## 2. Артефакты модели

DST поставляется как **bundle**, не как одиночный файл. Bundle включает:

```
dst-7b-r2.4/
├── manifest.json              ← версии, hashes, compatibility
├── weights.safetensors        ← veca model: backbone + heads (Q4 GPTQ или fp16)
├── tokenizer.json             ← BPE
├── tau_calibration.json       ← маппинг runtime signals → τ_observed (см. spec §1.4)
├── capsule_schema.json        ← N, D, slot layout, slot dtype
├── lookahead_config.json      ← acceptance thresholds, hierarchical layout
├── consistency_steps.json     ← когда 1-step vs 2-step denoise
└── runtime_contract.json      ← min/recommended VRAM, FLOP budget per token
```

`manifest.json`:
```json
{
  "model_id": "dst-7b",
  "version": "2.4.1",
  "tier": "standard",
  "capsule_schema_version": "1.0",
  "compatibility": {
    "min_runtime": "0.6.0",
    "min_capsule_schema": "1.0",
    "max_capsule_schema": "1.x"
  },
  "weights_sha256": "...",
  "signatures": [{"signer": "managed-ca", "sig": "..."}]
}
```

**Capsule schema versioning.** Любое изменение в N, D, slot layout или slot semantics — это major bump capsule schema. Workers с несовместимой schema получают 0% трафика. Это критично, иначе stale capsule после rollout новой версии разрушит сессии.

**Model registry** (managed):
- Принимает signed bundle от training pipeline.
- Раскатывает по worker pool через canary (5% → 25% → 100%).
- Откатывается на предыдущую версию при превышении error budget (см. §9).

---

## 3. Worker daemon

### 3.1 Жизненный цикл

```
boot → register → warm → ready → busy → drain → shutdown
```

- **boot:** запускается контейнер/binary, читает config.
- **register:** подключается к Gateway, отправляет attestation (GPU model, VRAM, network info, signed identity).
- **warm:** загружает выбранный bundle, прогревает CUDA kernels, делает self-test (генерирует 100 токенов на reference prompt и сверяет с oracle reference output).
- **ready:** подключается к work queue, ждёт сессии.
- **busy:** обслуживает 1..K параллельных сессий (см. §4.4 multi-session batching).
- **drain:** перестаёт принимать новые сессии, текущие checkpoint-ятся и мигрируют.
- **shutdown:** clean exit.

### 3.2 Что worker умеет

| Команда от Gateway | Действие |
|---|---|
| `attach(session_id, capsule, τ, prompt_suffix)` | Старт обслуживания сессии. Может прийти с пустой capsule (cold start) или загруженной (failover). |
| `step(session_id, user_input)` | Один шаг генерации. Вернуть accepted token chunk. |
| `checkpoint(session_id)` | Принудительно эмитировать capsule в State Store. |
| `detach(session_id, drain_to)` | Снять сессию, вернуть последнюю capsule. `drain_to` подсказывает, кому она пойдёт. |
| `health()` | Метрики: VRAM, GPU util, queue depth, p50/p99 latency. |
| `update(bundle_id)` | Загрузить новый model bundle, провести self-test, сообщить готовность. |

### 3.3 Worker config (consumer GPU, RTX 4090 пример)

```yaml
identity:
  worker_id: "auto"
  cert_path: /etc/dst/worker.crt
  key_path: /etc/dst/worker.key

resources:
  gpu: 0
  max_vram_mb: 22000        # оставляем 2GB на ОС/драйверы
  max_concurrent_sessions: 6
  max_context_tokens: 8192

network:
  gateway_addrs:
    - gw-eu-1.example.com:443
    - gw-eu-2.example.com:443
  prefer_lower_rtt: true
  max_upload_mbps: 30

model:
  bundle_dir: /var/lib/dst/bundles
  preferred_tier: "standard"
  quantization: "q4_gptq"

runtime:
  denoise_skip_threshold: 0.1
  multistep_threshold: 1.5
  checkpoint_cadence_tokens: 64
  checkpoint_cadence_seconds: 8
  telemetry_endpoint: telemetry.example.com:443
```

### 3.4 VRAM бюджет (RTX 4090, 24GB)

| Компонент | VRAM (Q4) |
|---|---:|
| DST 7B веса | 4.0 GB |
| KV-cache на window (64 tok × 6 sessions × 32 layers × 4096 dim × 2 × fp16) | ~3 GB |
| Capsule × 6 sessions (128 × 512 × fp16) | < 1 MB |
| Activations при inference (peak) | ~2.5 GB |
| Lookahead trees | ~500 MB |
| CUDA workspace | ~1 GB |
| **Итого** | **~11 GB** |

Запас в ~10 GB используется под (a) разворачивание quantized weights в compute kernels и (b) failover preload — заранее держим warm "next bundle" перед rollout.

На RTX 3060 (12GB) — те же 6 sessions не помещаются; **edge tier** (1.5B Q8) имеет:
- Веса: ~1.5 GB
- KV-cache: ~600 MB
- Activations: ~1 GB
- Capsule + lookahead: ~300 MB
- 4 sessions max: ~7 GB вписываемся.

---

## 4. Inference path

### 4.1 Сессионный жизненный цикл

```
1. Client → Gateway: open_session(prompt, options)
2. Gateway → Difficulty Estimator: difficulty(prompt) → tier
3. Gateway → Router: choose_worker(tier, region, load) → worker_W
4. Gateway → State Store: allocate(session_id) → empty capsule
5. Gateway → worker_W: attach(session_id, capsule=empty, τ=0, prefill=prompt)
6. worker_W: prefill prompt → c_after_prefill → (опционально) push checkpoint
7. Loop:
   Gateway → worker_W: step(session_id, ...) → accepted_tokens
   Gateway → Client: stream tokens
   periodically: worker_W → State Store: capsule checkpoint
8. End-of-stream OR client disconnect:
   Gateway → worker_W: detach(session_id)
```

### 4.2 Per-step протокол

Один step request — это **один remote round-trip**, возвращающий несколько токенов:

```protobuf
StepRequest {
  string session_id = 1;
  bytes user_input_tokens = 2;     // обычно пусто после prefill
  uint64 last_accepted_seq = 3;    // для idempotency / replay protection
  ControlFlags flags = 4;
}

StepResponse {
  bytes accepted_tokens = 1;       // 1..K токенов
  bytes structural_commits = 2;    // hierarchical lookahead intermediate, optional
  float self_uncertainty = 3;      // [0..1]
  uint64 capsule_seq = 4;          // bumped при чекпоинте
  optional bytes capsule_delta = 5;// если время checkpoint
  optional float tau_observed = 6; // worker's view of τ
  WorkerHealth health = 7;
}
```

QUIC поверх TLS 1.3, multiplexed streams для параллельных сессий на одном connection.

### 4.3 τ tracking на worker

Worker сам ведёт `τ_internal` для каждой сессии:

```python
def update_tau(session, event):
    base = session.tau_at_last_checkpoint  # обычно 0
    elapsed = now() - session.last_checkpoint_time
    drift = a * elapsed + b * session.compression_ratio
    session.tau_internal = clip(base + drift, 0, T_MAX)
```

Когда worker делает checkpoint, `tau_at_last_checkpoint = 0` и таймер обнуляется. Когда новый worker получает capsule из State Store, `tau_at_last_checkpoint` устанавливается по age of checkpoint.

### 4.4 Multi-session batching

На одном GPU worker обслуживает несколько сессий через **continuous batching** (vLLM-style):
- Все active sessions делят один forward через модель.
- Capsule для разных сессий упакованы в `batch_capsule[B, N, D]`.
- τ — per-session scalar, broadcast в AdaLN.
- KV-cache для sliding window — paged (PagedAttention).

Это критично для экономики P2P: без батчинга 4090 утилизирована на 15-20% при batch=1. С батчингом — 60-80%.

Hierarchical lookahead делает это сложнее (разные сессии могут оказаться на разной глубине дерева) — на Phase 1 lookahead отключаем при batch > 1, на Phase 2 решаем через speculative-friendly batching (как DraftBatch в vLLM).

### 4.5 Cold start vs warm continuation

**Cold start** (новая сессия):
1. Worker получает prompt, делает prefill.
2. Prefill — это T forward-проходов, по одному на токен промпта (но в batched режиме — один pass с causal mask).
3. После prefill capsule содержит "сжатую" промпта (через slot updates).
4. Сразу пушим первый checkpoint в State Store (даёт failover safety).
5. Начинаем generation.

**Warm continuation** (failover на нового воркера):
1. Worker получает `(capsule_chk, τ_estimated_from_age)`.
2. Денойзит capsule (1-step, поскольку τ обычно средний).
3. Иногда нужно "догнать" токены, сгенерированные между checkpoint и failover. Если их ≤ K (обычно 5-50) — replay через быстрый prefill на denoised state.
4. Начинаем generation.

**Cold fallback** (capsule утеряна или несовместима):
1. Worker получает полный prompt + всю историю диалога.
2. Полный prefill — это freeze 1-3 секунды, который виден клиенту.
3. Это плохой UX, но redundancy: если bundle несовместим или State Store недоступен, fallback срабатывает автоматически.

### 4.6 Когда денойзить (политика)

```
if τ_internal < 0.1:
    skip_denoise()              # лишний forward не нужен
elif τ_internal < 1.5:
    denoise_1step()             # обычный режим
else:
    denoise_2step()              # после длительного offline или после failover
```

Замер: на 4090 один denoise step добавляет ~3-4ms к token generation. Skip threshold экономит ~25% времени на стационарных сессиях.

### 4.7 Lookahead acceptance loop

Worker генерирует hierarchical chunk:
```
[t+1 token, t+1 struct_class,
 t+2 token | t+1, t+2 struct,
 t+3 token | t+2, t+3 struct]
```

Gateway получает chunk, **верификации не делает** (gateway тупой). Worker сам же на следующем step проверяет, что accepted предсказания консистентны (auto-verification): если struct_class на t+1 не совпал с продолжением, отбрасывает t+2/t+3 и пересчитывает.

Альтернатива: верификация на **другом** воркере того же tier (anti-byzantine, см. §6.3). Включается probabilistically.

---

## 5. State Store

### 5.1 Что хранит

Per session:
- Last K capsule checkpoints (delta-compressed chain).
- Token stream (для cold fallback).
- Session metadata: model version, tier, region, owner, timestamps.
- Audit log: какой worker когда что сделал.

### 5.2 Размер

Для 1M активных сессий, capsule 7B = 5MB до сжатия, 500KB после delta + quantization, K=4 хранимых versions:
```
1M × 4 × 500KB = 2 TB hot tier
```

Hot tier — Redis cluster или DynamoDB-style fast KV. Cold tier (закрытые сессии за 24-72 часа) — S3.

### 5.3 Consistency

Capsule chain — append-only с monotonic sequence numbers. Конкурентные writes от разных воркеров (например, во время failover) разрешаются last-writer-wins по `(worker_id, sequence_no)`, но Gateway routes одну сессию строго одному worker за раз, так что конфликтов в норме не бывает.

### 5.4 Privacy

- Capsule зашифрована per-session ключом, выдаваемым Gateway.
- Worker получает ключ только пока сессия attached.
- При detach ключ revoke-ается на этом воркере.
- Долговременные ключи Gateway хранит в KMS.

Это не делает capsule полностью private (Gateway видит всё), но защищает от "worker украл capsule после disconnect".

---

## 6. Trust & integrity

### 6.1 Worker attestation

При registration worker предоставляет:
- Подписанный сертификат (issued нашим CA при onboarding).
- TPM/SGX attestation, если доступно (опционально, не блокирующее).
- Hardware fingerprint (GPU UUID, MAC, etc).
- Software hash (binary + bundle hashes).

Gateway отвергает неподписанных или sanity-failed воркеров. Подозрительные воркеры попадают в quarantine pool с reduced traffic.

### 6.2 Capsule integrity

Каждый capsule checkpoint несёт:
```
{
  capsule_bytes: ...,
  prev_checkpoint_hash: sha256(...),
  worker_signature: sign(capsule_bytes || prev_hash, worker_key),
  sequence_no: N
}
```

Это **chain commit**: rewrite старой capsule сломает chain hash и будет отвергнут State Store. Защита от malicious rewind.

### 6.3 Probabilistic verifier sampling

Gateway randomly выбирает 1-2% токенов и переотправляет state на oracle (или другого worker того же tier) для верификации:
- Если KL(p_worker || p_oracle) > threshold — сессия мигрирует, worker получает negative reputation.
- Накопление negative repute → blacklist.

Это байт-уровневая дисциплина, не cryptography. Защищает от deliberate quality attacks (worker, который выдаёт мусор).

### 6.4 Что нельзя обеспечить

- **Privacy от worker.** Worker видит plaintext конверсейшн. Без homomorphic encryption (которое для трансформеров не работает по compute budget) это не лечится. Mitigation: классификация сессий — sensitive trafic не идёт на untrusted P2P, только на managed tier.
- **Worker не лжёт про generation.** Verifier sampling даёт probabilistic guarantee, не deterministic.

---

## 7. Routing & quality scaling

### 7.1 Difficulty estimator

Лёгкая модель (~50M) на Gateway, вход — prompt embedding + metadata, выход — оценка difficulty score [0..1]. На основе score:

| Difficulty | Tier | Notes |
|---:|---|---|
| 0.0–0.3 | edge (1.5B) | fast path, low cost |
| 0.3–0.7 | standard (7B) | default |
| 0.7–1.0 | oracle (14B/32B managed) | escalation |

Update DE на feedback loop: если standard tier генерация имеет высокий self-uncertainty или низкую human satisfaction → DE учится поднимать score таких prompts.

### 7.2 Token-level escalation

Не только request-level, но и **token-level**: если worker возвращает `self_uncertainty > 0.6` на нескольких подряд токенах, Gateway временно вызывает oracle на следующий chunk, пока uncertainty не упадёт. Это дешевле чем полный oracle re-run.

### 7.3 Best-of-N в idle время

Если cluster utilization < 50%, gateway opportunistically запускает 2-3 параллельных worker для одного запроса с разным seed. Reranking через oracle (или uncertainty heuristic). Бесплатное улучшение качества за счёт неиспользуемой capacity.

### 7.4 Specialist routing

В пул входят specialist воркеры (code/math/safety). Они не маршрутизируются на target generation, а вызываются как post-hoc critics на span-уровне:
- Generation идёт стандартно.
- Параллельно span сannotation модель разметила code/math/safety spans.
- Specialist re-validates accepted tokens в этих spans.
- Если specialist disagree → token rejection и regenerate.

---

## 8. Развёртывание новых версий модели

### 8.1 Pipeline

```
training cluster → eval suite → bundle build → registry signing → rollout
```

Gates:
1. **Eval pass:** PPL/quality regression checks vs previous version.
2. **Capsule schema check:** schema bumped только если breaking; migration plan если major.
3. **Compute budget check:** не превышает FLOP/token budget.
4. **Manual approval** для major version.

### 8.2 Canary rollout

Worker pool segmented:
- 5% canary workers получают новый bundle, обслуживают 5% трафика.
- Telemetry collected: PPL, latency, error rate, user satisfaction proxies (если есть).
- Если canary ok 24h → 25% → 100%.
- Авто-откат при > X% error rate increase.

### 8.3 Capsule migration при breaking schema bump

Сценарий: schema 1.0 → 2.0 несовместима.
- Все active sessions при detection migration **drain** через cold fallback (полный re-prefill).
- Workers разделены на 1.0 pool и 2.0 pool, gateway routes по версии.
- 1.0 pool постепенно сжимается до нуля по мере drain.
- Workers получают 2.0 bundle и регистрируются в 2.0 pool.
- ETA migration зависит от timeout сессий (обычно ≤ 30 min).

Это плохой UX, поэтому breaking schema bump планируется заранее и совмещается с maintenance windows.

### 8.4 Worker-side update

Worker получает `update(bundle_id)`:
1. Загружает bundle в фоновом disk cache.
2. После активной сессии (или drain) переключается на новый bundle.
3. Делает self-test.
4. Сообщает Gateway о готовности к 2.0 трафику.

Двух bundles одновременно в VRAM не держим — это слишком дорого. На диске — да, на GPU — только активный.

---

## 9. Observability & SRE

### 9.1 Метрики (минимум)

**Per worker:**
- `dst_session_active` (gauge)
- `dst_token_latency_seconds` (histogram, p50/p95/p99)
- `dst_lookahead_acceptance_rate` (histogram)
- `dst_tau_observed` (histogram)
- `dst_denoise_invocations_total` (counter, by skip/1step/2step)
- `dst_checkpoint_emit_total` (counter)
- `dst_self_uncertainty` (histogram)
- `dst_failover_received_total` (counter, by reason)
- `dst_vram_used_bytes`, `dst_gpu_utilization`

**Per gateway:**
- `dst_session_open_total`, `dst_session_close_total`
- `dst_router_decisions_total` (by tier, by reason)
- `dst_oracle_escalation_total`
- `dst_worker_blacklist_total`
- `dst_capsule_chain_hash_mismatch_total` (security signal)

**Per state store:**
- `dst_capsule_bytes_in / out`
- `dst_checkpoint_age_seconds` (histogram)
- `dst_capsule_chain_depth` (per session)

### 9.2 SLO целевые

| SLO | Цель |
|---|---|
| Token generation p50 latency | ≤ 200 ms на чанк (≥ 3 tok/chunk) |
| Token generation p99 latency | ≤ 800 ms на чанк |
| Session failover invisible (no client-visible freeze) rate | ≥ 99% |
| Lookahead acceptance rate | ≥ 60% |
| Verifier sampling disagreement rate | ≤ 1% (выше — расследование) |
| Capsule chain hash mismatches | 0 (любой incident — security) |

### 9.3 Error budget rollout gate

Если за 7 дней:
- > 0.5% sessions experienced cold fallback freeze, OR
- > 2% verifier sampling disagreement, OR
- > 0.1% capsule schema mismatch

→ rollout новых версий freeze, расследование.

---

## 10. Что в этом плане **не** будет в MVP

Скоп MVP (Phase 1) — минимум, чтобы запустить end-to-end на 0.5B + 50 workers:

- Single-tier routing (только standard tier, без edge/oracle).
- No specialist workers.
- No verifier sampling (есть chain hash, но не KL re-check).
- No best-of-N.
- No multi-region gateway.
- No QUIC, начнём с TLS+gRPC (HTTP/2).
- No TPM attestation (только cert-based).
- Capsule storage — Postgres + S3, без отдельного Redis (через год при нагрузке заменим).

Это даёт развёртывание за ~3 месяца после G0.

Phase 2 (production hardening):
- Multi-tier + escalation.
- QUIC.
- Best-of-N + verifier sampling.
- Multi-region.
- Specialist critics.

Phase 3 (scale):
- 1000+ workers.
- Capsule schema migration tooling.
- Federated training updates (опционально, требует много отдельного дизайна).

---

## 11. Прикидка экономики (sanity check)

Не финансовая модель, sanity check.

**Один RTX 4090 worker, standard tier, 6 concurrent sessions:**
- 15 tok/s × 6 sessions = 90 tok/s output.
- При 24/7 utilization: 7.8M tokens/day.
- Обработать 1M токенов на datacenter inference (Anthropic Claude pricing aside) — порядка $1-5.
- Worker hardware ammort + electricity ≈ $1-2/day для 4090.

При 50% utilization economics 4090 worker сравнима с datacenter inference при условии:
- 80%+ session uptime (мало failover-cold-restart-сессий).
- 60%+ lookahead acceptance.
- < 5% oracle escalation.

Если эти условия не держатся, P2P экономика проигрывает datacenter — и оправдание остаётся только в "available compute that wouldn't exist otherwise" (idle consumer GPU) и "cost-distributed across users".

Это не финансовая модель, это grounding для product decisions.

---

## 12. Открытые инженерные риски

1. **Capsule schema lock-in.** После запуска изменить N или D без breaking — почти невозможно. Версионирование частично закрывает, но active sessions страдают. **Mitigation:** заложить запас в Phase 2 schema, не делать N/D ровно по minimum.

2. **Worker churn.** Consumer GPU = нерегулярная доступность (пользователь играет в игру → worker offline). При высоком churn rate failover нагрузка превышает batched capacity. **Mitigation:** churn-aware routing (long sessions → stable workers); short bursty запросы → опportunistic workers.

3. **NAT traversal.** Большинство consumer connections за NAT/CGNAT. Worker не может accept inbound. **Mitigation:** worker инициирует outbound long-lived connection к Gateway (через QUIC/HTTP2), gateway pushes work через эту connection. Это уже стандарт для tunneled architectures.

4. **Bandwidth asymmetry.** Consumer upload часто 10-50 Mbps. Capsule transmit 5MB × раз в 50 токенов × 6 sessions = ~3 MB/s = 24 Mbps — на грани. **Mitigation:** delta-checkpoint (см. §4.7 в spec); адаптивная cadence (если upload загружен → реже checkpoint).

5. **Denoise overhead in batched setting.** Multi-session batch + denoise step = AdaLN с разными τ broadcast, что ломает некоторые fused kernels. Возможно, придётся сегрегировать сессии по τ-bucket или платить overhead. **Mitigation:** реализовать AdaLN-batched kernel; в худшем случае держать 2-3 батча по τ-bucket.

6. **Quantization × τ.** Q4 модель может не сохранить denoising качество на high τ — denoising требует точности на shape capsule manifold. **Mitigation:** Phase 0 эксперимент с Q4 на DST. Если плохо — fp8 или mixed precision (denoiser в fp16, остальное в Q4).

---

**Owners:** TBD.
**Last updated:** 2026-04-28.
