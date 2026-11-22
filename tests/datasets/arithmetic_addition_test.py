import pytest
import torch
from trainite.datasets.arithmetic_addition import (
    ArithmeticAdditionDataset,
    PromptCompletionTransform,
)
from trainite.preprocessors.char_tokenizer import CharTokenizer


def test_pool_size_and_uniqueness():
    dataset = ArithmeticAdditionDataset(size=500)

    assert len(dataset) == 500
    assert len({item["source"] for item in dataset}) == 500


def test_sample_format():
    dataset = ArithmeticAdditionDataset(size=100)
    item = dataset[0]

    assert set(item) == {
        "source",
        "target",
        "original_source",
        "original_target",
        "operand_lengths",
        "digit_order",
    }
    assert item["digit_order"] == "msd_first"
    a, b = item["source"].split("+")
    assert item["target"] == str(int(a) + int(b))
    assert item["original_source"] == item["source"]
    assert item["original_target"] == item["target"]
    assert isinstance(item["operand_lengths"], tuple)


def test_reverse_digits_format():
    dataset = ArithmeticAdditionDataset(size=100, reverse_digits=True)
    item = dataset[0]

    assert item["digit_order"] == "lsd_first"
    a, b = item["source"].split("+")
    assert a == item["original_source"].split("+")[0][::-1]
    assert b == item["original_source"].split("+")[1][::-1]
    assert item["target"] == item["original_target"][::-1]


def test_pad_to_width_format():
    dataset = ArithmeticAdditionDataset(size=100, min_digits=2, max_digits=4, pad_to_width=True)
    item = dataset[0]

    assert item["digit_order"] == "msd_first"
    a, b = item["source"].split("+")
    assert len(a) == len(b) == 4
    assert len(item["target"]) == 5
    assert int(a) + int(b) == int(item["target"])


def test_pad_to_width_and_reverse_digits_are_exclusive():
    with pytest.raises(ValueError, match="separate formats"):
        ArithmeticAdditionDataset(size=10, reverse_digits=True, pad_to_width=True)


def test_digit_range_respected():
    dataset = ArithmeticAdditionDataset(size=200, min_digits=3, max_digits=5)

    for item in dataset:
        digits_a, digits_b = item["operand_lengths"]
        assert 3 <= digits_a <= 5
        assert 3 <= digits_b <= 5
        a, b = item["source"].split("+")
        assert len(a) == digits_a
        assert len(b) == digits_b


def test_same_seed_is_reproducible():
    a = ArithmeticAdditionDataset(size=200, seed=7)
    b = ArithmeticAdditionDataset(size=200, seed=7)

    assert [item["source"] for item in a] == [item["source"] for item in b]


def test_different_seed_gives_different_pool():
    a = ArithmeticAdditionDataset(size=200, seed=1)
    b = ArithmeticAdditionDataset(size=200, seed=2)

    assert [item["source"] for item in a] != [item["source"] for item in b]


def test_impossible_request_raises_instead_of_hanging():
    # 2-digit operands give only 90 * 90 = 8100 unique problems.
    with pytest.raises(ValueError, match="only 8100 pairs exist"):
        ArithmeticAdditionDataset(size=10_000, min_digits=2, max_digits=2)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="Invalid operand digit range"):
        ArithmeticAdditionDataset(size=10, min_digits=5, max_digits=3)
    with pytest.raises(ValueError, match="size must be positive"):
        ArithmeticAdditionDataset(size=0)


def test_prompt_completion_transform():
    tokenizer = CharTokenizer()
    transform = PromptCompletionTransform(tokenizer=tokenizer, ignore_index=-100)

    dataset = ArithmeticAdditionDataset(size=1, reverse_digits=True)
    raw = dataset[0]
    item = transform(raw)

    assert item.source == raw["source"]
    assert item.target == raw["target"]
    assert item.train_input_ids.dtype == torch.long
    assert len(item.train_input_ids) == len(item.train_label_ids)
    assert item.train_input_ids[0].item() == tokenizer.bos_token_id
    assert item.train_label_ids[-1].item() == tokenizer.eos_token_id
    # Only the target digits and EOS contribute to the loss.
    assert (item.train_label_ids == -100).sum().item() == len(item.eval_input_ids) - 1
    assert item.original_source == raw["original_source"]
    assert item.operand_lengths == raw["operand_lengths"]
    assert item.digit_order == "lsd_first"


def test_library_config_splits_are_disjoint_and_load():
    """The shipped library defaults must not put the same problem in two splits."""
    from trainite.config.datasets import ArithmeticAdditionDataConfig
    from trainite.shared.utils import build_dataloaders

    config = ArithmeticAdditionDataConfig()
    train_loader, val_loader, test_loader = build_dataloaders(config, CharTokenizer(), seed=config.dataset.seed)

    train, val, test = train_loader.dataset, val_loader.dataset, test_loader.dataset
    # One pool, three disjoint index ranges.
    assert train.dataset is val.dataset is test.dataset
    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices).isdisjoint(test.indices)
    assert set(val.indices).isdisjoint(test.indices)
    assert (len(train), len(val), len(test)) == (8000, 1000, 1000)
