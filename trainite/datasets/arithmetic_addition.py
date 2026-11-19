"""Synthetic dataset for character-level arithmetic addition."""

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate addition problems with disjoint train/validation splits.

    For ``split="train"`` and ``split="val"``, the samples are drawn from a
    single pool of ``train_size + val_size`` unique problems and sliced, so the
    two splits are disjoint by construction (they must share the same seed,
    digit range, and pool sizes).

    For ``split="test"``, ``size`` samples are generated independently and
    disjointness from train/val is *not* enforced; pass a distinct seed and/or
    digit range to guarantee it.
    """

    def __init__(
        self,
        size: int | None = None,
        min_digits: int = 2,
        max_digits: int = 4,
        seed: int = 42,
        split: str = "train",
        train_size: int = 10000,
        val_size: int = 1000,
    ):
        if min_digits <= 0 or max_digits < min_digits:
            raise ValueError("Invalid operand digit range")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if train_size <= 0 or val_size <= 0:
            raise ValueError("train_size and val_size must be positive")

        expected_size = {
            "train": train_size,
            "val": val_size,
            "test": size or 1000,
        }[split]
        if size is None:
            size = expected_size
        elif size != expected_size and split != "test":
            raise ValueError(f"size must be {expected_size} for split={split!r}")
        if size <= 0:
            raise ValueError("size must be positive")

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed
        self.split = split

        if split == "test":
            self.samples = self._generate_samples(size, seed)
        else:
            expected_size = train_size if split == "train" else val_size
            if size != expected_size:
                raise ValueError(f"size must be {expected_size} for split={split!r}")

            all_samples = self._generate_samples(train_size + val_size, seed)

            if split == "train":
                self.samples = all_samples[:train_size]
            else:
                self.samples = all_samples[train_size:]

    def _generate_samples(self, count, seed):
        rng = random.Random(seed)
        samples = []
        seen_sources = set()

        while len(samples) < count:
            digits_a = rng.randint(self.min_digits, self.max_digits)
            digits_b = rng.randint(self.min_digits, self.max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)

            source = f"{a}+{b}"

            if source in seen_sources:
                continue

            seen_sources.add(source)
            samples.append(
                {
                    "source": source,
                    "target": str(a + b),
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DatapointModel(BaseModel):
    """Contract for a causal-LM transformed item: training tensors + eval prompt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    target: str
    train_input_ids: torch.Tensor
    train_label_ids: torch.Tensor
    attention_mask: torch.Tensor
    eval_input_ids: torch.Tensor


class PromptCompletionTransform:
    """Convert a raw arithmetic sample into training tensors.

    The model is trained with next-token prediction on the target string and is
    evaluated by generating the target from the prompt alone.  All prompt
    positions (including the trailing SEP) are masked out of the loss, so only
    target digits and EOS contribute.
    """

    def __init__(self, tokenizer: Any, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def build_prompt(self, sample: dict[str, str]) -> list[int]:
        """Token ids for the generation prompt: [bos] + source + [sep] (no target)."""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        source_tokens = self.tokenizer(sample["source"], add_special_tokens=False)["input_ids"]
        return [bos] + source_tokens + [sep]

    def __call__(self, sample: dict[str, str]) -> DatapointModel:
        source = sample["source"]
        target = sample["target"]

        eos = self.tokenizer.eos_token_id

        prompt_ids = self.build_prompt(sample)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Full sequence: prompt + target tokens + EOS
        combined_input_ids = prompt_ids + target_tokens + [eos]

        # input_ids is the sequence shifted right by 1 (teacher-forcing style)
        input_ids = torch.tensor(combined_input_ids[:-1], dtype=torch.long)
        # labels is the sequence shifted left by 1 (next-token prediction targets)
        labels = torch.tensor(combined_input_ids[1:], dtype=torch.long)

        # Mask the prompt portion of the (shifted) labels. labels = combined[1:],
        # so the prompt spans len(prompt_ids) - 1 leading positions.
        labels[: len(prompt_ids) - 1] = self.ignore_index

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=attention_mask,
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        )
