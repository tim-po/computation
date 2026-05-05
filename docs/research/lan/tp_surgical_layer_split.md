---
title: "TP-Surgical: Методология разбиения слоёв (Layer Partitioning)"
date: 2026-05-04
author: TBD
version: v1.1
status: research-draft
stream: lan
tags: [lan, tp-surgical, layer-split, gqa, activation-profiling, methodology, distillation]
---

# TP-Surgical: Методология разбиения слоёв

Как правильно разрезать трансформер (Qwen3 и любую другую модель) для TP-Surgical. Покрывает:
- какие компоненты делить, какие нет
- как обрабатывать GQA
- как определить точки CCA через активационную статистику
- таблица параметров для каждого размера Qwen3

Связанный документ: [`tp_surgical.md`](tp_surgical.md)

---

## 1. Инвентаризация: что режем, что нет

### 1.1 Не трогаем (реплицировано на все GPU)

| Компонент | Размер | Причина |
|---|---|---|
| `embed_tokens` | V × H | lookup по vocab; маршрутизация по GPU дороже, чем репликация |
| `lm_head` | H × V | tied с embed_tokens; разрезать = All-Gather перед каждым logit |
| Все `RMSNorm` | H или 1 | element-wise по hidden dim, крошечные параметры |
| RoPE | — | нет весов; позиция вычисляется per-head без коммуникации |

**Вывод:** всё, что не зависит от head-индекса или промежуточной размерности MLP — остаётся целым.

### 1.2 Режем по голосам (attention)

Это column-parallel split — каждый GPU владеет непрерывным диапазоном голов.

```
Q proj: [H × (n_heads · head_dim)]
K proj: [H × (n_kv_heads · head_dim)]
V proj: [H × (n_kv_heads · head_dim)]
O proj: [(n_heads · head_dim) × H]   ← row-parallel, reduce после
```

GPU g (из N) получает головы `[g·n_heads/N … (g+1)·n_heads/N)`.

**O-proj** использует row-parallel: каждый GPU считает свою часть выхода, затем при CCA суммируются.

### 1.3 Режем MLP (SwiGLU)

Column-parallel по intermediate dimension `I`:

```
gate_proj: [H × I]  → GPU g берёт [H × I/N]
up_proj:   [H × I]  → GPU g берёт [H × I/N]
down_proj: [I × H]  → GPU g берёт [I/N × H]  (row-parallel, reduce после)
```

MLP и Attention независимо параллелятся; CCA синхронизирует оба одновременно.

### 1.4 Только на ведущем GPU

- CCA correction weights (~31M параметров): принимает конкатенат скрытых состояний со всех GPU, выдаёт скорректированный вектор
- Logit sampling и KV-cache management (пока нет WAN-failover)

---

## 2. GQA: правило делимости

Qwen3 использует Grouped Query Attention: `n_kv_heads < n_heads`. Это создаёт ограничение на число GPU.

**Правило:** `N_gpu` должно делить `n_kv_heads`.

При нарушении: K/V головы нельзя равномерно распределить → нужна репликация или неравные шарды (ведёт к load imbalance).

### Таблица допустимых конфигураций для Qwen3

| Model | n_heads | n_kv_heads | Допустимые N_gpu |
|---|---:|---:|---|
| Qwen3-0.6B | 16 | 8 | **1, 2, 4, 8** |
| Qwen3-1.7B | 16 | 8 | **1, 2, 4, 8** |
| Qwen3-4B | 32 | 8 | **1, 2, 4, 8** |
| Qwen3-8B | 32 | 8 | **1, 2, 4, 8** |
| Qwen3-14B | 40 | 8 | **1, 2, 4, 8** |
| Qwen3-32B | 64 | 8 | **1, 2, 4, 8** |

> Для любого Qwen3 `n_kv_heads = 8` → всегда можно взять 2, 4 или 8 GPU. 16+ требует K/V-репликации.

---

## 3. Рекомендуемые параметры разбиения (Qwen3)

Параметры берутся из `config.json` модели. Ниже — ориентировочные значения.

| Model | n_layers | H | head_dim | I (MLP) | Shard/GPU (2×) |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 1024 | 64 | ~2816 | ~145 MB |
| Qwen3-1.7B | 28 | 2048 | 128 | ~5504 | ~530 MB |
| Qwen3-4B | 36 | 2560 | 80 | ~6912 | ~1.2 GB |
| Qwen3-8B | 36 | 4096 | 128 | ~11008 | ~2.4 GB |
| Qwen3-14B | 40 | 5120 | 128 | ~13824 | ~4.2 GB |
| Qwen3-32B | 64 | 5120 | 80 | ~27648 | ~9.4 GB |

> Shard/GPU = примерный объём весов на один GPU при 2-GPU split в bfloat16.

---

## 4. Активационная методология: где ставить CCA

Текущий K=4 — эвристика. Правильный способ — data-driven placement через профилирование.

### 4.1 Протокол

**Шаг 1: Тестовая выборка**

50–100 промптов, равномерно из 4 типов:
- Диалог (короткий, < 512 токенов)
- Код (Python/Rust, ~256 токенов)
- Длинный контекст (> 1024 токенов)
- Математика / рассуждения

**Шаг 2: Baseline сбор на полной модели**

Записать `H_l` — hidden states после каждого слоя `l` для каждого токена.

```python
hooks = []
hidden_states = {}
for l, layer in enumerate(model.model.layers):
    def hook_fn(module, input, output, l=l):
        hidden_states[l] = output[0].detach()  # [batch, seq, H]
    hooks.append(layer.register_forward_hook(hook_fn))
model(**inputs)
for h in hooks: h.remove()
```

**Шаг 3: Симуляция расщепления**

Для каждого слоя `l` — оценить, как накапливается дивергенция если синхронизации не было с последней CCA:

```python
# split simulation: GPU 0 видит только heads [0, n/2), GPU 1 видит [n/2, n)
# из-за row-parallel O-proj их вклад не суммируется → разница в h_l
# cumulative divergence δ(l):
delta = {}
h_approx = H[0].clone()  # начинаем от последней CCA
for l in range(last_cca, max_layer):
    h_approx = simulate_split_step(h_approx, l, n_gpu=2)
    delta[l] = frobenius_relative(h_approx, H[l])
    # frobenius_relative = ||h_approx - H[l]||_F / ||H[l]||_F
```

**Шаг 4: Выбор точек CCA**

Вставить CCA перед слоем `l` если `δ(l) > τ`:
- `τ = 0.03` — агрессивный (частые CCA, больше синхронизации, меньше ошибка)
- `τ = 0.05` — сбалансированный (рекомендуется как стартовая точка)
- `τ = 0.08` — редкие CCA (быстрее, но выше ошибка)

**Шаг 5: Проверка качества**

После расстановки CCA — быстрый proxy: BPB на 1000 примерах Wikitext-2.

```
целевой gap: < +5% vs full model BPB
допустимый: < +10%
требует пересмотра: > +10%
```

### 4.2 Ожидаемая форма кривой δ(l)

Из анализа трансформеров известно:

```
Дивергенция δ(l) по слоям:
  ▲
  │                          ___...
  │              ___....----
  │      ___....
  │ ___-
  │
  └─────────────────────────────► l
     early   middle           late
```

- **Ранние слои (l < 20%):** δ растёт медленно — токены ещё не взаимодействуют глубоко
- **Средние слои (20–75%):** δ растёт быстрее — основная семантическая обработка, головы специализируются
- **Поздние слои (75–100%):** δ часто плато — предсказание следующего токена локально, меньше зависимость от других голов

**Вывод:** неравномерный schedule (редкие CCA вначале, частые в середине) теоретически лучше фиксированного K. Проверить на данных.

### 4.3 Baseline расписание vs data-driven (пример для 28 слоёв)

| Подход | CCA точки | Число RTT | Комментарий |
|---|---|---:|---|
| K=4 равномерно | 4, 8, 12, 16, 20, 24 | 6 | текущий baseline |
| τ=0.05 data-driven | зависит от профиля | 5–8 | ожидается лучше по quality/RTT |
| Неравномерный (ручной) | 5, 10, 13, 16, 19, 22, 25 | 7 | больше CCA в середине |

---

## 5. Что делать с эмбеддингами

### Вариант А: Полная репликация (рекомендуется)

`embed_tokens` реплицирован на все GPU. Каждый GPU выполняет свой lookup.

- Pros: никакой коммуникации на embedding step
- Cons: 300–600 MB VRAM дублируется для 7B+ модели

Для 0.6B–4B: VRAM embedding = 0.6–1.5 GB — приемлемо при 24 GB/GPU.

### Вариант Б: Vocab Parallel (для больших моделей)

Разрезать `embed_tokens` по словарю: GPU g владеет токенами `[g·V/N … (g+1)·V/N)`.
После lookup — All-Gather скрытых состояний перед первым слоем.

- Pros: экономия VRAM ~V·H/N bytes
- Cons: All-Gather на каждый шаг = +1 RTT в начале, усложняет код

**Рекомендация:** репликация для ≤ 8B, vocab-parallel рассмотреть для 14B+.

---

## 6. Обобщение на любую модель

Алгоритм применим к любому decoder-only трансформеру:

```
1. Прочитать config.json:
   - n_layers, hidden_size, num_heads, num_key_value_heads, intermediate_size

2. Проверить: N_gpu | num_key_value_heads
   - Нет: уменьшить N_gpu до ближайшего делителя

3. Вычислить shard size:
   - attn_shard = 2 * hidden * (num_heads/N + num_kv_heads/N) * head_dim  [Q+K = col; V = col]
   - o_proj_shard = (num_heads/N) * head_dim * hidden
   - mlp_shard = hidden * (intermediate/N) * 3  [gate+up+down]
   - per_layer = attn_shard + o_proj_shard + mlp_shard
   - total_shard = per_layer * n_layers + embed (if replicated)

4. Запустить activation profiling (Шаг 2-3 из §4.1)

5. Разместить CCA по τ = 0.05

6. Обучить CCA correction weights (31M → масштабируется к H)

7. Validate: BPB gap < 5%
```

---

## 7. Чек-лист перед запуском эксперимента

- [ ] `N_gpu | n_kv_heads` выполняется
- [ ] Shard size укладывается в VRAM с запасом (KV-cache занимает до 30-40% VRAM)
- [ ] Activation profiling запущен на репрезентативном датасете (≥ 4 типа промптов)
- [ ] Кривая δ(l) построена и CCA точки выбраны
- [ ] BPB gap < 5% (или задокументировано почему допустим gap > 5%)
- [ ] Измерен tok/s при реальном RTT LAN (~5ms), а не на localhost

---

## 8. Результаты на Qwen3.5-0.8B (MPS, 2026-05-04)

Первый прогон activation profiling. Скрипты: [`experiments/01_baseline.py`](../../experiments/01_baseline.py), [`experiments/02_activation_profiling.py`](../../experiments/02_activation_profiling.py)

### Архитектурные особенности

| Параметр | Значение |
|---|---|
| Архитектура | Гибридная: `[linear_attention × 3, full_attention × 1] × 6` |
| Full attention @ | слои 3, 7, 11, 15, 19, 23 |
| n_heads / n_kv_heads | 8 / 2 → **max TP-split = 2 GPU** для full attention |
| linear_num_key_heads | 16 → max TP-split = 16 GPU для linear attention |
| MTP встроен | `mtp_num_hidden_layers=1` — lookahead head уже есть! |
| vocab_size | 248,320 |

### Baseline inference (M4 MPS, single device)

| Метрика | Значение |
|---|---|
| Среднее tok/s | **19.7 tok/s** (min=15.4, max=25.6) |
| Загрузка модели | 5.9s (из кэша) |
| Fallback | flash-linear-attention не установлен → torch fallback |

### Residual magnitude: ключевые наблюдения

- **Слой 0 аномален:** residual=2.55, все остальные avg ≈ 1.2 → всегда ставить CCA после слоя 0
- **Пик середины:** слои 14-15 (residual 1.60/1.65) — критичная зона
- **Тихий хвост:** слои 20-22 (residual ~0.98-1.06) — CCA здесь не обязательна
- **Конец:** слой 23 (1.54) — последний full_attention, нужна CCA

### Head contribution balance (N_gpu=2)

Слои 19 и 23 — head imbalance > 5%: GPU_0 доминирует (57-60%). При сплите GPU_1 теряет непропорционально много сигнала именно в конце сети.

### CCA schedule comparison

| Schedule | Точки CCA | RTT/токен | Теоретич. tok/s |
|---|---|---:|---:|
| K=4 uniform | [3, 7, 11, 15, 19, 23] | 6 | **33** |
| Data-driven top-8 | [0, 3, 6, 7, 11, 14, 15, 23] | 8 | 25 |
| **Рекомендуемый** | **[0, 3, 7, 11, 15, 19, 23]** | **7** | **28** |

Рекомендуемый schedule: K=4 как основа (покрывает full_attention слои) + layer 0 (аномальный residual). 7 RTT × 5ms = 35ms/tok → **28 tok/s**.

---

## 9. Открытые вопросы

1. **Linear attention сплит:** `linear_num_key_heads=16` — linear attention слои можно split до 16 GPU. Как работает CCA для linear (state-based) vs full (softmax) attention?

2. **Head imbalance нарастает к концу:** слои 19-23 показывают 56/44 и 60/40. Структурная особенность или зависит от промптов?

3. **Перепрофилирование после SFT:** активации изменятся после SFT. Нужен ли повторный profiling?

4. **Correction weight scaling:** сейчас 31M параметров для 0.5B. Для 0.8B / 7B нужно ли масштабировать?

5. **flash-linear-attention:** после установки FLA tok/s вырастет. Перебенчмаркить baseline.
