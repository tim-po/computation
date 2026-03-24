"""Data loading: WikiText-103 with GPT-2 tokenizer, packed into fixed-length sequences."""

import torch
from torch.utils.data import Dataset, DataLoader


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
