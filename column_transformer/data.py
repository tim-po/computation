"""Data loading: WikiText-103 and FineWeb-Edu with GPT-2 tokenizer, packed into fixed-length sequences."""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class PackedTextDataset(Dataset):
    """Tokenizes and packs text into fixed-length chunks for LM training."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Pack into non-overlapping chunks of seq_len + 1 (input + target)
        n_tokens = len(token_ids)
        n_chunks = n_tokens // (seq_len + 1)
        self.data = torch.tensor(
            token_ids[: n_chunks * (seq_len + 1)], dtype=torch.long
        ).view(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]  # input, target


def load_wikitext(seq_len: int = 512, batch_size: int = 32, num_workers: int = 2):
    """Load WikiText-103 and return train/val DataLoaders."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_split(split_name: str) -> list[int]:
        texts = dataset[split_name]["text"]
        # Filter empty lines and tokenize in batches
        all_ids = []
        batch = []
        for text in texts:
            text = text.strip()
            if text:
                batch.append(text)
            if len(batch) >= 1000:
                encoded = tokenizer(batch, add_special_tokens=False)
                for ids in encoded["input_ids"]:
                    all_ids.extend(ids)
                batch = []
        if batch:
            encoded = tokenizer(batch, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                all_ids.extend(ids)
        return all_ids

    print("Tokenizing train split...")
    train_ids = tokenize_split("train")
    print(f"  Train: {len(train_ids):,} tokens -> {len(train_ids) // (seq_len + 1):,} sequences")

    print("Tokenizing validation split...")
    val_ids = tokenize_split("validation")
    print(f"  Val: {len(val_ids):,} tokens -> {len(val_ids) // (seq_len + 1):,} sequences")

    train_dataset = PackedTextDataset(train_ids, seq_len)
    val_dataset = PackedTextDataset(val_ids, seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, val_loader


class StreamingPackedDataset(IterableDataset):
    """Streams from HuggingFace dataset, tokenizes on-the-fly, packs into fixed-length chunks."""

    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        token_buffer = []
        chunk_len = self.seq_len + 1
        for example in self.hf_dataset:
            text = example.get("text", "").strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)
            while len(token_buffer) >= chunk_len:
                chunk = torch.tensor(token_buffer[:chunk_len], dtype=torch.long)
                token_buffer = token_buffer[chunk_len:]
                yield chunk[:-1], chunk[1:]


def load_fineweb_edu(seq_len: int = 1024, batch_size: int = 16, val_docs: int = 10000):
    """Load FineWeb-Edu (sample-10BT) in streaming mode and return train/val DataLoaders."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading FineWeb-Edu dataset (streaming)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Materialize a small validation set first
    print(f"Materializing validation set ({val_docs:,} docs)...")
    val_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    val_ids = []
    count = 0
    for example in val_stream:
        if count >= val_docs:
            break
        text = example.get("text", "").strip()
        if text:
            val_ids.extend(tokenizer.encode(text, add_special_tokens=False))
            count += 1
    print(f"  Val: {len(val_ids):,} tokens -> {len(val_ids) // (seq_len + 1):,} sequences")

    val_dataset = PackedTextDataset(val_ids, seq_len)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Streaming train set (skips val docs to avoid overlap)
    print("Setting up streaming train loader...")
    train_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    train_stream = train_stream.skip(val_docs).shuffle(buffer_size=10000, seed=42)
    train_dataset = StreamingPackedDataset(train_stream, tokenizer, seq_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader
