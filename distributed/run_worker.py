"""CLI entry point for a distributed inference worker.

Usage:
    python -m distributed.run_worker \
        --model h100_col8_drop25 \
        --shard shards/worker_1.pt \
        --column 1 \
        --coordinator localhost:9000
"""

import argparse
import asyncio
import logging
import torch

from column_transformer.config import EXPERIMENTS, ColumnConfigV2
from .worker import DistributedWorker

logger = logging.getLogger(__name__)


async def run_worker(args):
    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2), f"{args.model} is not a column model"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    host, port = args.coordinator.split(":")
    port = int(port)

    worker = DistributedWorker(
        column_idx=args.column,
        config=config,
        shard_path=args.shard,
        coordinator_host=host,
        coordinator_port=port,
        device=device,
    )

    print(f"Worker {args.column}: connecting to {host}:{port}...")
    await worker.connect_and_run(n_forward=args.n_forward)
    print(f"Worker {args.column}: done")


def main():
    parser = argparse.ArgumentParser(description="Distributed inference worker")
    parser.add_argument("--model", required=True, help="Model config name")
    parser.add_argument("--shard", required=True, help="Path to worker shard checkpoint")
    parser.add_argument("--column", type=int, required=True, help="Column index this worker runs")
    parser.add_argument("--coordinator", default="localhost:9000",
                        help="Coordinator address (host:port)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-forward", type=int, default=1000000,
                        help="Max forward passes before exiting (default: run until shutdown)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    asyncio.run(run_worker(args))


if __name__ == "__main__":
    main()
