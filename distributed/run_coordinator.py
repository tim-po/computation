"""CLI entry point for the distributed inference coordinator.

Usage:
    python -m distributed.run_coordinator \
        --model h100_col8_drop25 \
        --checkpoint results_h100/h100_col8_drop25_best.pt \
        --port 9000 \
        --wait-for 3 \
        --prompt "The future of artificial intelligence" \
        --max-tokens 100
"""

import argparse
import asyncio
import logging
import time
import torch

from column_transformer.config import EXPERIMENTS, ColumnConfigV2
from .coordinator import DistributedCoordinator

logger = logging.getLogger(__name__)


async def run_inference(args):
    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2), f"{args.model} is not a column model"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    coordinator = DistributedCoordinator(
        config=config,
        checkpoint_path=args.checkpoint,
        host="0.0.0.0",
        port=args.port,
        device=device,
        coordinator_column=args.coordinator_column,
        merge_timeout=args.merge_timeout,
    )

    await coordinator.start_server()
    print(f"Coordinator listening on port {args.port}")
    print(f"Waiting for {args.wait_for} workers...")

    await coordinator.wait_for_workers(
        min_workers=args.wait_for,
        timeout=args.wait_timeout,
    )
    print(f"Active columns: {coordinator.active_columns} "
          f"({coordinator.n_active}/{config.n_columns})")

    # Tokenize prompt
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if args.prompt:
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    else:
        # Default test input
        input_ids = torch.randint(0, config.vocab_size, (1, 64), device=device)

    # Run forward pass(es)
    if args.generate:
        print(f"\nGenerating {args.max_tokens} tokens...")
        generated = input_ids.clone()
        t0 = time.time()

        for step in range(args.max_tokens):
            # Truncate to max_seq_len
            context = generated[:, -config.max_seq_len:]
            logits = await coordinator.forward(context)

            # Greedy or top-k sampling
            next_logits = logits[:, -1, :]
            if args.temperature > 0:
                next_logits = next_logits / args.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Decode and print
            token_str = tokenizer.decode(next_token[0])
            print(token_str, end="", flush=True)

        dt = time.time() - t0
        print(f"\n\n--- {args.max_tokens} tokens in {dt:.1f}s "
              f"({args.max_tokens/dt:.1f} tok/s) ---")
    else:
        # Single forward pass benchmark
        print(f"\nRunning {args.n_forward} forward pass(es)...")
        times = []
        for i in range(args.n_forward):
            t0 = time.time()
            logits = await coordinator.forward(input_ids)
            dt = time.time() - t0
            times.append(dt)
            print(f"  Pass {i+1}: {dt*1000:.1f}ms, logits shape: {logits.shape}")

        if len(times) > 1:
            avg = sum(times[1:]) / len(times[1:])  # Skip warmup
            print(f"\n  Average (excl. warmup): {avg*1000:.1f}ms")
            tokens = input_ids.shape[1]
            print(f"  Throughput: {tokens/avg:.0f} tokens/s")

    await coordinator.shutdown_workers()


def main():
    parser = argparse.ArgumentParser(description="Distributed inference coordinator")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--checkpoint", required=True, help="Full model checkpoint path")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--coordinator-column", type=int, default=0)
    parser.add_argument("--wait-for", type=int, default=1,
                        help="Minimum workers to wait for before starting")
    parser.add_argument("--wait-timeout", type=float, default=120.0)
    parser.add_argument("--merge-timeout", type=float, default=30.0)

    # Inference mode
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--generate", action="store_true", help="Autoregressive generation mode")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-forward", type=int, default=5,
                        help="Number of forward passes for benchmark mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
