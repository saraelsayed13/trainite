"""Synthetic dataset for character-level arithmetic addition.

The samples follow the data format used by the Abacus paper
(McLeish et al., 2024, https://arxiv.org/abs/2405.17399): numbers are written
least-significant digit first (LSB-first / reversed).  Reversal is what makes
digit columns line up across operands and, more importantly, what makes the
Abacus column id of an already-generated token independent of how many more
digits will be generated.

The same dataset implementation and the same deterministic splits are used by
both experiments (RoPE + AdamW baseline and Abacus + AdamW).  Set
``reverse_digits: true`` in both configs to keep the comparison fair.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
import random

from torch.utils.data import Dataset


class ArithmeticAdditionDataset(Dataset):
    """Generate a flat, deterministic pool of addition problems.

    The dataset does not decide train/val/test membership.  It generates
    ``size`` unique problems from ``seed``; the caller splits the pool for
    train/val and builds a separate pool with a different digit range and seed
    for the out-of-distribution test set.

    Each item is a dictionary with:

    * ``source`` / ``target``: the strings fed to the tokenizer (LSB-first when
      ``reverse_digits=True``, fixed-width zero-padded when ``pad_to_width=True``).
    * ``original_source`` / ``original_target``: the conventional MSB-first
      strings, kept for readable logs and reports.
    * ``operand_lengths``: ``(digits_a, digits_b)``, used for per-length
      exact-match reporting.
    * ``digit_order``: ``"lsd_first"`` or ``"msd_first"``.
    """

    def __init__(
        self,
        size: int = 10000,
        min_digits: int = 2,
        max_digits: int = 4,
        seed: int = 42,
        reverse_digits: bool = False,
        pad_to_width: bool = False,
    ):
        if min_digits <= 0 or max_digits < min_digits:
            raise ValueError("Invalid operand digit range")
        if pad_to_width and reverse_digits:
            raise ValueError("pad_to_width and reverse_digits are separate formats; pick one")
        if size <= 0:
            raise ValueError("size must be positive")

        self.size = size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.seed = seed
        self.reverse_digits = reverse_digits
        self.pad_to_width = pad_to_width
        self.samples = self._generate_samples(size, seed)

    def _generate_samples(self, count: int, seed: int) -> list[dict[str, object]]:
        rng = random.Random(seed)
        samples: list[dict[str, object]] = []
        seen_sources: set[str] = set()

        possible_values = 10**self.max_digits - 10 ** (self.min_digits - 1)
        if count > possible_values**2:
            raise ValueError(
                f"Cannot generate {count} unique problems with "
                f"{self.min_digits}-{self.max_digits} digits; only "
                f"{possible_values**2} pairs exist."
            )

        max_attempts = max(10_000, count * 100)
        attempts = 0
        while len(samples) < count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Gave up after {max_attempts} attempts with {len(samples)}/{count} unique problems."
                )

            digits_a = rng.randint(self.min_digits, self.max_digits)
            digits_b = rng.randint(self.min_digits, self.max_digits)

            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1)

            original_source = f"{a}+{b}"
            original_target = str(a + b)
            if self.pad_to_width:
                # Zero-pad experiment: every operand gets the same number of
                # columns, and the answer gets one extra column for the carry.
                width = self.max_digits
                source = f"{str(a).zfill(width)}+{str(b).zfill(width)}"
                target = original_target.zfill(width + 1)
                digit_order = "msd_first"
            elif self.reverse_digits:
                source = f"{str(a)[::-1]}+{str(b)[::-1]}"
                target = original_target[::-1]
                digit_order = "lsd_first"
            else:
                source = original_source
                target = original_target
                digit_order = "msd_first"

            if source in seen_sources:
                continue

            seen_sources.add(source)
            samples.append(
                {
                    "source": source,
                    "target": target,
                    "original_source": original_source,
                    "original_target": original_target,
                    "operand_lengths": (digits_a, digits_b),
                    "digit_order": digit_order,
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
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

    # Optional metadata used by evaluation and readable logging.  They are not
    # needed by the model or the training loop.
    original_source: str | None = None
    original_target: str | None = None
    operand_lengths: tuple[int, int] | None = None
    digit_order: str = "lsd_first"


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

    def build_prompt(self, sample: dict[str, Any]) -> list[int]:
        """Token ids for the generation prompt: [BOS] + source + [SEP]."""
        bos = self.tokenizer.bos_token_id
        sep = self.tokenizer.sep_token_id
        source_tokens = self.tokenizer(sample["source"], add_special_tokens=False)["input_ids"]
        return [bos] + source_tokens + [sep]

    def __call__(self, sample: dict[str, Any]) -> DatapointModel:
        source = sample["source"]
        target = sample["target"]

        eos = self.tokenizer.eos_token_id
        prompt_ids = self.build_prompt(sample)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Full sequence: prompt + target tokens + EOS.  input_ids is the sequence
        # shifted right by one and labels are shifted left by one, so position t
        # predicts token t+1.
        combined_input_ids = prompt_ids + target_tokens + [eos]
        input_ids = torch.tensor(combined_input_ids[:-1], dtype=torch.long)
        labels = torch.tensor(combined_input_ids[1:], dtype=torch.long)

        # labels = combined[1:]; the prompt spans len(prompt_ids) - 1 positions.
        labels[: len(prompt_ids) - 1] = self.ignore_index
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        operand_lengths = sample.get("operand_lengths")
        if operand_lengths is not None:
            operand_lengths = tuple(operand_lengths)
            if len(operand_lengths) != 2:
                operand_lengths = None

        return DatapointModel(
            source=source,
            target=target,
            train_input_ids=input_ids,
            train_label_ids=labels,
            attention_mask=attention_mask,
            eval_input_ids=torch.tensor(prompt_ids, dtype=torch.long),
            original_source=sample.get("original_source"),
            original_target=sample.get("original_target"),
            operand_lengths=operand_lengths,
            digit_order=sample.get("digit_order", "lsd_first"),
        )
