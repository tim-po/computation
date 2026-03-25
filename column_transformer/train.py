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
):
    device = get_device()
    print(f"\nTraining {model_name} on {device}")
    print(f"  Parameters: {count_parameters(model):,}")

    model = model.to(device)
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
    while step < max_steps:
        epoch += 1
        for input_ids, targets in train_loader:
            if step >= max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Cosine schedule with warmup
            if step < warmup_steps:
                cur_lr = lr * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Forward
            logits = model(input_ids)  # [B, T, vocab]
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            step += 1
            train_losses.append((step, loss.item()))

            # Logging
            if step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", cur_lr, step)
                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (max_steps - step) / steps_per_sec
                print(
                    f"  [{model_name}] step {step}/{max_steps} | "
                    f"loss {loss.item():.4f} | lr {cur_lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s | ETA {eta/60:.0f}min"
                )

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
