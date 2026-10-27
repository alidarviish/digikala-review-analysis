"""Pieces of Transformer fine-tuning that do not need PyTorch, so they can be tested anywhere."""

import numpy as np

IGNORE = -100  # label of a review that has no label for a head; the loss skips it


def fixed_sample(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Positions of a random sample, in their original order, the same on every machine."""
    if size >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size, replace=False))


def label_codes(values, labels: list[str]) -> np.ndarray:
    """Class number of each label, or IGNORE where the label is missing."""
    code = {label: i for i, label in enumerate(labels)}
    result = np.full(len(values), IGNORE, dtype=np.int64)
    for i, value in enumerate(values):
        if value is not None and value == value:  # value == value is False for NaN
            result[i] = code[value]
    return result


def linear_warmup(step: int, total: int, warmup: int) -> float:
    """Learning-rate factor: rises linearly over the warmup steps, then falls linearly to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def build_model(torch, name: str, dropout: float, heads: list[str], classes: int = 3):
    """One encoder with a linear head per task on the first token. torch is passed in, so this
    module can be imported (and tested) where PyTorch is not installed."""
    from transformers import AutoModel

    class TwoHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            size = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(dropout)
            self.heads = torch.nn.ModuleDict({head: torch.nn.Linear(size, classes) for head in heads})

        def forward(self, input_ids, attention_mask):
            first = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            first = self.dropout(first)
            return {head: layer(first) for head, layer in self.heads.items()}

    return TwoHeads()


def batches(torch, tokenizer, encoded, order, size, labels=None):
    """Padded batches of the given rows, in the given order, with their labels if any."""
    for start in range(0, len(order), size):
        rows = order[start : start + size]
        padded = tokenizer.pad({"input_ids": [encoded[i] for i in rows]}, return_tensors="pt")
        batch = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"]}
        if labels is not None:
            batch["labels"] = {head: torch.as_tensor(codes[rows]) for head, codes in labels.items()}
        yield batch
