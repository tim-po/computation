# Document Index

*Single source of navigation. Update whenever a document is added, moved, or its status changes.*

**Last updated:** 2026-05-05 | Stream overview: [`STREAMS.md`](STREAMS.md)

---

## 🟢 LAN Stream — текущий фокус (RTT ~5ms)

### Research

| Document | Date | Author | Version | Status |
|---|---|---|---|---|
| [TP-Surgical: Tensor-Parallel Inference (LAN)](research/lan/tp_surgical.md) | 2026-04-26 | TBD | v1.1 | stable |
| [TP-Surgical: Методология разбиения слоёв](research/lan/tp_surgical_layer_split.md) | 2026-05-04 | TBD | v1.1 | research-draft |
| [TP-Surgical: Roadmap до деплоя](research/lan/roadmap_tp_surgical.md) | 2026-05-05 | TBD | v1.0 | living-document |

### Architecture

| Document | Date | Author | Version | Status |
|---|---|---|---|---|
| [Архитектурное кладбище](architecture/failed_architectures.md) | 2026-05-04 | TBD | v2.0 | living-document |

### Updates

| Document | Date | Author | Version | Status |
|---|---|---|---|---|
| [Cofounder Brief 2026-04-26](updates/brief_2026-04-26.md) | 2026-04-26 | Timophey | v1.0 | final |
| [Cofounder Update 2026-05-01: Surgical-1.5B + Skeleton-Fill](updates/cofounder_update_2026-05-01.md) | 2026-05-01 | Timophey | v1.0 | final |

---

## 🔵 WAN Stream — будущий скейлинг (RTT ~100ms)

### Research

| Document | Date | Author | Version | Status |
|---|---|---|---|---|
| [DST Research Spec](research/wan/dst-research-spec.md) | 2026-04-28 | TBD | v1.0 | research-draft |
| [DST Engineering: Deployment & Inference](research/wan/dst-engineering.md) | 2026-04-28 | TBD | v1.0 | research-draft |
| [Time-Traveling Columns (TTC)](research/wan/ttc_scaling.md) | 2026-04-24 | TBD | v1.0 | draft |

### Architecture

| Document | Date | Author | Version | Status |
|---|---|---|---|---|
| [Decentralized LLM — Синтез исследования](architecture/decentralized-llm-synthesis.md) | 2026-04-29 | TBD | v1.0 | living-document |
| [Summary-State Transformer для WAN](architecture/columnar-decentralized-llm.md) | 2026-04-25 | TBD | v1.0 | stable |

---

## Diagrams

| File | Description |
|---|---|
| [architecture.mmd](diagrams/architecture.mmd) / [.png](diagrams/architecture.png) | General architecture overview |
| [wan_stateful_architecture.mmd](diagrams/wan_stateful_architecture.mmd) / [.png](diagrams/wan_stateful_architecture.png) | WAN stateful architecture |
| [wan_stateless_capsule_architecture.mmd](diagrams/wan_stateless_capsule_architecture.mmd) / [.png](diagrams/wan_stateless_capsule_architecture.png) | WAN stateless capsule architecture |
| [tp_surgical.mmd](diagrams/tp_surgical.mmd) / [.png](diagrams/tp_surgical.png) | TP-Surgical (LAN) |
| [ttc_architecture.mmd](diagrams/ttc_architecture.mmd) / [.png](diagrams/ttc_architecture.png) | Time-Traveling Columns (WAN) |
| [speculative.mmd](diagrams/speculative.mmd) / [.png](diagrams/speculative.png) | Speculative decoding |

## Drafts

*Work in progress. Not yet promoted to a stream.*

| Document | Date | Author | Status | Description |
|---|---|---|---|---|
| *(empty)* | | | | |

---

## Status glossary

| Status | Meaning |
|---|---|
| `draft` | Early idea, may change significantly |
| `research-draft` | Formal spec under active investigation |
| `stable` | Architecture decision made, not expected to change |
| `living-document` | Updated continuously as the project evolves |
| `final` | Immutable snapshot (briefs/updates) |
| `archived` | Superseded or abandoned |
