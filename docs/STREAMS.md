---
title: "Research Streams: LAN vs WAN"
date: 2026-05-04
author: TBD
version: v1.0
status: living-document
tags: [streams, lan, wan, architecture, roadmap]
---

# Research Streams: LAN vs WAN

Проект развивается по двум независимым исследовательским стримам, отличающимся одним ключевым параметром — **задержкой между оркестратором и нодами**.

| Параметр | LAN | WAN |
|---|---|---|
| RTT оркестратор → нода | **~5 ms** | **~100 ms** |
| Типичная топология | Локальная сеть, NVLink, PCIe | Публичный интернет, P2P |
| Типичное железо | 2-8 GPU в одном rack/комнате | Consumer GPU по всему миру |
| **Текущий статус** | **🟢 Активный фокус** | **🔵 Будущий скейлинг** |

---

## Почему разные стримы?

20× разница в RTT полностью меняет какие архитектуры работают:

- **Per-token sync операции** на WAN стоят 100ms каждая → 1-3 tok/s (неприемлемо)
- **Те же операции** на LAN стоят 5ms → 33+ tok/s (отлично)

Многие подходы, которые "провалились" при исследовании WAN, **полностью работоспособны на LAN**.

---

## Матрица совместимости архитектур

| Архитектура | WAN (100ms) | LAN (5ms) | Причина различия |
|---|:---:|:---:|---|
| **TP-Surgical (CCA каждые 4 слоя)** | ❌ | ✅ | 6 RTT × 5ms = 30ms/tok vs 600ms/tok |
| **Pipeline Parallelism (слои по нодам)** | ❌ | ✅ | feedback loop стоит 5ms, не 100ms |
| **Standard Tensor Parallelism** | ❌ | ⚠️ | 24 RTT × 5ms = 120ms/tok (~8 tok/s) |
| **MoE с экспертами на разных нодах** | ❌ | ⚠️ | 24 layers × 5ms = 120ms/tok, маргинально |
| **Capsule sharding между нодами** | ❌ | ✅ | 1 RTT per chunk × 5ms = приемлемо |
| **Async pipeline + rollback** | ❌ | ⚠️ | rollback window = ~1 tok на LAN |
| **Per-token full logits voting** | ❌ | ⚠️ | bandwidth issue (800 Mbps), не latency |
| **Naive ensemble** | ❌ | ❌ | математическая проблема, не latency |
| **Stateful KV-cache thrashing** | ❌ | ❌ | структурная проблема |
| **Idealized spec decoding** | ❌ | ❌ | acceptance rate — не RTT проблема |
| **Summary-State Transformer (capsule)** | ✅ | ✅ | разработан для WAN, на LAN — оверхед |
| **DST (Diffusion-State Transformer)** | ✅ | ✅ | разработан для WAN |
| **TTC (Time-Traveling Columns)** | ✅ | ✅ | lookahead скрывает любой RTT |

**Легенда:** ✅ работает | ⚠️ маргинально / требует оптимизации | ❌ не работает

---

## LAN стрим — текущий фокус 🟢

**Гипотеза:** Tensor-Parallel-Surgical Retrofit (TP-Surgical) на локальной сети даёт production-quality inference уже сегодня, без необходимости разрабатывать новую capsule-архитектуру.

**Что валидировано (2026-04-26):**
- 2× RTX 3090 по LAN → **23.5 tok/s** (2.5× faster than single GPU)
- Worker shard = **145 MB** (Qwen 0.5B)
- Quality recovery после 1000 шагов дообучения: **93%** от vanilla Qwen

**Текущие эксперименты:**
- Surgical-1.5B: validated end-to-end с SFT, BPB gap vs Qwen = +8.1%
- FIM (Fill-in-the-Middle) smoke test: Skeleton+Fill как ответ на bandwidth проблему

**Ключевые документы:**
- [`research/lan/tp_surgical.md`](research/lan/tp_surgical.md) — архитектура и результаты TP-Surgical
- [`updates/brief_2026-04-26.md`](updates/brief_2026-04-26.md) — первый валидированный brief
- [`updates/cofounder_update_2026-05-01.md`](updates/cofounder_update_2026-05-01.md) — Surgical-1.5B финальный eval + Skeleton+Fill

**Open questions (LAN):**
1. Насколько хорошо масштабируется TP-Surgical за пределы 2 GPU на LAN?
2. Skeleton+Fill — правильный ли это ответ на bandwidth bottleneck при 4+ GPU?
3. Что происходит с качеством при 7B+ при LAN TP-Surgical?

---

## WAN стрим — будущий скейлинг 🔵

**Гипотеза:** После того как LAN inference работает стабильно, P2P/WAN расширение даёт access to distributed idle GPU compute без необходимости владеть железом.

**Основная ставка — DST (Diffusion-State Transformer):**
Capsule state эволюционирует как denoising diffusion. Failover, jitter, corruption — одна ось τ. 1 RTT возвращает chunk токенов через lookahead heads, скрывая 100ms пинг.

**Decision gate G0** (нужно пройти перед WAN скейлингом):
- DST PPL within 5% of baseline — must pass
- Recovery ≤ 5 tokens at τ=1.0 — must pass
- Failover quality at Δ=50 within 10% of clean — must pass

**Ключевые документы:**
- [`research/wan/dst-research-spec.md`](research/wan/dst-research-spec.md) — полная research-спека DST
- [`research/wan/dst-engineering.md`](research/wan/dst-engineering.md) — deployment & inference
- [`research/wan/ttc_scaling.md`](research/wan/ttc_scaling.md) — Time-Traveling Columns
- [`architecture/columnar-decentralized-llm.md`](architecture/columnar-decentralized-llm.md) — базовая WAN архитектура

---

## Путь от LAN к WAN

```
Phase 0 (сейчас)
  LAN: TP-Surgical 1.5B validated ✅
  LAN: Skeleton+Fill smoke test 🔄
  LAN: Scale to 4+ GPU, fix bandwidth

Phase 1 (~3-4 мес)
  LAN: Production 7B на 4-8 GPU кластерах
  WAN: DST Phase 0 gate G0 experiment (параллельно)

Phase 2 (~6 мес)
  WAN: MVP — 50 P2P workers, если G0 пройден
  LAN: Оптимизация, multi-tenant, enterprise edition

Phase 3
  WAN: 1000+ workers, full P2P network
  LAN→WAN: Micro-pod bridge (LAN кластеры как логические WAN-воркеры)
```

**Micro-pod bridge** — LAN-кластер (2-8 GPU на NVLink/PCIe) выглядит как один логический WAN-воркер. Это позволяет скейлить LAN-работу в WAN сеть без изменения WAN-протокола.
