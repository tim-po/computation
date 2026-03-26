"""Training loop shared by all model variants."""

import math
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .evaluate import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    max_steps: int = 20000,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 500,
    max_grad_norm: float = 1.0,
    eval_every: int = 500,
    log_dir: str = "runs",
    model_name: str = "model",
    save_dir: str = "checkpoints",
    grad_accum_steps: int = 1,
    use_bf16: bool = False,
    use_compile: bool = False,
):
    device = get_device()
    print(f"\nTraining {model_name} on {device}")
    print(f"  Parameters: {count_parameters(model):,}")
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps")

    # Mixed precision setup
    amp_enabled = use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    if amp_enabled:
        print(f"  Mixed precision: bfloat16")

    model = model.to(device)

    # torch.compile for extra speed (H100 benefits significantly)
    if use_compile and hasattr(torch, "compile"):
        print(f"  torch.compile: enabled")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    writer = SummaryWriter(log_dir=f"{log_dir}/{model_name}")
    loss_fn = nn.CrossEntropyLoss()

    # Training state
    step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_perplexities = []
    epoch = 0
    t_start = time.time()

    model.train()
    micro_step = 0  # counts micro-batches within an accumulation window
    accum_loss = 0.0
    optimizer.zero_grad()

    while step < max_steps:
        epoch += 1
        for input_ids, targets in train_loader:
            if step >= max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward with optional mixed precision
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(input_ids)  # [B, T, vocab]
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / grad_accum_steps  # scale for accumulation

            # bf16 doesn't need GradScaler (unlike fp16)
            loss.backward()

            accum_loss += loss.item()
            micro_step += 1

            if micro_step < grad_accum_steps:
                continue

            # --- Optimizer step (every grad_accum_steps micro-batches) ---
            step += 1
            micro_step = 0

            # Cosine schedule with warmup
            if step < warmup_steps:
                cur_lr = lr * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append((step, accum_loss))

            # Logging
            if step % 50 == 0:
                writer.add_scalar("train/loss", accum_loss, step)
                writer.add_scalar("train/lr", cur_lr, step)
                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (max_steps - step) / steps_per_sec
                print(
                    f"  [{model_name}] step {step}/{max_steps} | "
                    f"loss {accum_loss:.4f} | lr {cur_lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | ETA {eta/60:.0f}min"
                )

            accum_loss = 0.0

            # Evaluation
            if step % eval_every == 0 or step == max_steps:
                val_loss, val_ppl = evaluate(model, val_loader, device, loss_fn)
                val_perplexities.append((step, val_ppl))
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/perplexity", val_ppl, step)
                print(
                    f"  [{model_name}] EVAL step {step} | "
                    f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pt")

                model.train()

    writer.close()
    total_time = time.time() - t_start
    print(f"  [{model_name}] Finished in {total_time/60:.1f} min | best val_loss {best_val_loss:.4f}")

    n_params = count_parameters(model)
    # Move model to CPU to free GPU memory before next model
    model.cpu()
    del optimizer
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "params": n_params,
        "train_losses": train_losses,
        "val_perplexities": val_perplexities,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
    }
