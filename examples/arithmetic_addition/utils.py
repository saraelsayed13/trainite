import importlib
import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import yaml
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    create_lr_scheduler_with_warmup,
)
from ignite.handlers.clearml_logger import ClearMLSaver
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.handlers.logger_utils import setup_wandb_logging
from ignite.handlers.param_scheduler import ParamScheduler
from ignite.utils import setup_logger
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, random_split

from config import (
    DataConfigBase,
    ProjectConfig,
)
from dataset_impl.transformed import TransformedDataset

T = TypeVar("T", bound=BaseModel)


def get_target(target_path: str) -> Any:
    if not target_path:
        raise ValueError("The '_target_' key must not be empty.")

    try:
        # Split 'module.submodule.ClassName' into 'module.submodule' and 'ClassName'
        module_path, symbol_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_symbol = getattr(module, symbol_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate target '{target_path}': {e}") from e

    return target_symbol


def _inject_if_accepted(target_symbol: Any, *, allow_var_kwargs: bool = True, **candidates: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(target_symbol)
        if allow_var_kwargs and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return candidates
        return {k: v for k, v in candidates.items() if k in sig.parameters}
    except Exception:
        return candidates


def instantiate(config: BaseModel, **kwargs) -> Any:
    if isinstance(config, BaseModel):
        config_dict = config.model_dump(by_alias=True, polymorphic_serialization=True)
    else:
        raise ValueError("Config must be an instance of BaseModel")

    if not isinstance(config_dict, dict) or "_target_" not in config_dict:
        raise ValueError("Config must be a dict containing a '_target_' key")

    # Work on a copy to avoid mutating the original config data
    params = config_dict.copy()
    target_path = params.pop("_target_")

    target_symbol = get_target(target_path)

    final_kwargs = {**params, **kwargs}
    final_kwargs = _inject_if_accepted(target_symbol, **final_kwargs)

    return target_symbol(**final_kwargs)


def dump_yaml(data: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def dump_config(config: BaseModel, path: str | Path) -> None:
    data = config.model_dump(by_alias=True, polymorphic_serialization=True)
    dump_yaml(data, path)


def load_config(path: str | Path, config_cls: type[T]) -> T:
    raw_conf = OmegaConf.load(path)
    return config_cls.model_validate(OmegaConf.to_container(raw_conf, resolve=True))


# Builds the model based on the provided configuration, tokenizer, and device.
def build_model(model_config: Any, device: str | torch.device, **kwargs) -> nn.Module:
    target_symbol = get_target(model_config.target)
    kwargs = _inject_if_accepted(target_symbol, **kwargs)
    model = instantiate(model_config, **kwargs)
    model.to(device)
    return model


def build_dataset(dataset_config: Any, transform_config: Any, tokenizer: Any) -> Dataset:
    ds = get_target(dataset_config.target)
    dataset = instantiate(
        dataset_config,
        **_inject_if_accepted(
            ds,
            allow_var_kwargs=False,
            preprocessor=tokenizer,
            tokenizer=tokenizer,
        ),
    )
    transform = None
    if transform_config is not None:
        tf = get_target(transform_config.target)
        transform = instantiate(
            transform_config, **_inject_if_accepted(tf, preprocessor=tokenizer, tokenizer=tokenizer)
        )
    return TransformedDataset(dataset, transform)


def create_dataloader(
    dataset: Dataset,
    dl_config: Any,
    tokenizer: Any,
    shuffle: bool | None = None,
    collate_fn_target: str | None = None,
) -> DataLoader:
    dl_kwargs = dl_config.model_dump(exclude={"shuffle"})
    if shuffle is None:
        shuffle = getattr(dl_config, "shuffle", False)
    collate_fn = None
    if collate_fn_target:
        target_symbol = get_target(collate_fn_target)
        if isinstance(target_symbol, type):
            collate_fn = target_symbol(tokenizer=tokenizer)
        else:
            collate_fn = target_symbol
    return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **dl_kwargs)


def build_split_datasets(
    data_config: DataConfigBase,
    tokenizer: Any,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Return ``(train, val, test)`` with train/val drawn from one shared pool.

    Train and validation share a single deterministic pool so they can never
    overlap: ``random_split`` assigns disjoint indices.  The test dataset is
    built separately because it is out-of-distribution (different digit
    range/seed) and therefore cannot come from the same pool.
    """

    def _params(split_config: Any) -> dict[str, Any]:
        params = split_config.dataset.model_dump()
        params.pop("size", None)
        return params

    if _params(data_config.train) != _params(data_config.val):
        raise ValueError("train and val datasets must share the same parameters (except size)")

    n_train = data_config.train.dataset.size
    n_val = data_config.val.dataset.size
    pool_config = data_config.train.dataset.model_copy(update={"size": n_train + n_val})
    pool = build_dataset(pool_config, data_config.train.transform, tokenizer)

    train_ds, val_ds = random_split(
        pool,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    test_ds = None
    if data_config.test is not None:
        test_ds = build_dataset(data_config.test.dataset, data_config.test.transform, tokenizer)

    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_config: DataConfigBase,
    tokenizer: Any,
    seed: int,
    collate_fn_target: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train/val/test loaders using :func:`build_split_datasets`."""
    train_ds, val_ds, test_ds = build_split_datasets(data_config, tokenizer, seed)

    train_loader = create_dataloader(
        train_ds,
        data_config.train.dataloader,
        tokenizer,
        shuffle=True,
        collate_fn_target=collate_fn_target,
    )
    val_loader = create_dataloader(
        val_ds,
        data_config.val.dataloader,
        tokenizer,
        shuffle=False,
        collate_fn_target=collate_fn_target,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = create_dataloader(
            test_ds,
            data_config.test.dataloader,
            tokenizer,
            shuffle=False,
            collate_fn_target=collate_fn_target,
        )

    return train_loader, val_loader, test_loader


def make_run_dir(output_config: Any) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = output_config.run_name
    run_dir = Path(output_config.root) / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_run_dir(config: ProjectConfig) -> Path:
    run_dir = make_run_dir(config.output)
    dump_config(config, run_dir / "config.yaml")
    return run_dir


def attach_lr_scheduler(
    engine: Engine,
    optimizer: Any,
    total_iters: int,
    peak_lr: float,
) -> None:
    # Reserve at least 2 iterations for warm-up even for very short training runs
    warmup_iters = max(2, int(0.1 * total_iters))
    linear_decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_iters - warmup_iters,
    )
    scheduler: ParamScheduler = create_lr_scheduler_with_warmup(
        linear_decay,
        warmup_start_value=0.0,
        warmup_end_value=peak_lr,
        warmup_duration=warmup_iters,
    )
    engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)


def attach_early_stopping(
    val_evaluator: Engine,
    trainer_engine: Engine,
    patience: int,
) -> None:
    early_stopping = EarlyStopping(
        patience=patience,
        score_function=lambda engine: engine.state.metrics["loss"],
        trainer=trainer_engine,
        min_delta=0.0,
        mode="min",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)


def setup_training_checkpointing(
    engine: Engine,
    to_save: dict[str, Any],
    save_handler: DiskSaver | ClearMLSaver,
) -> Checkpoint:
    last_checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=save_handler,
        filename_prefix="last",
        n_saved=1,
        global_step_transform=lambda *_: engine.state.iteration,
        filename_pattern="{filename_prefix}.{ext}",
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)
    return last_checkpoint


def setup_best_model_checkpoint(
    engine: Engine,
    val_evaluator: Engine,
    to_save: dict[str, Any],
    save_handler: DiskSaver | ClearMLSaver,
    score_function: Callable,
    score_name: str,
) -> Checkpoint:
    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=save_handler,
        filename_prefix="best",
        score_function=score_function,
        score_name=score_name,
        n_saved=1,
        global_step_transform=lambda *_: engine.state.iteration,
        filename_pattern="{filename_prefix}.{ext}",
    )
    val_evaluator.add_event_handler(Events.COMPLETED, checkpoint)
    return checkpoint


def setup_console_logger(
    run_dir: Path,
    engine: Engine,
    log_every_steps: int,
    optimizer: Any,
) -> logging.Logger:
    logger = setup_logger(
        "trainer",
        level=logging.INFO,
        filepath=str(run_dir / "output.log") if run_dir else None,
        reset=True,
    )
    train_fb_logger = FBResearchLogger(logger=logger, show_output=True)
    train_fb_logger.attach(
        engine,
        name="Train",
        every=log_every_steps,
        optimizer=optimizer,
        output_transform=lambda output: {"loss": output["loss"].item()},
    )
    return logger


def resolve_wandb_mode(mode: str) -> str:
    env_mode = os.environ.get("WANDB_MODE")
    if env_mode:
        return env_mode
    if mode == "auto":
        return "online" if os.environ.get("WANDB_API_KEY") else "offline"
    return mode


def setup_wandb_logging_safe(
    config: Any,
    run_dir: Path,
    logging_kwargs: dict[str, Any],
) -> Any | None:
    if not getattr(config, "wandb", None) or not config.wandb.enabled:
        return None

    mode = resolve_wandb_mode(config.wandb.mode)
    if mode == "disabled":
        return None

    try:
        import wandb  # noqa: F401
    except ImportError:
        logging.getLogger("trainer").warning("wandb is not installed; W&B logging disabled.")
        return None

    init_kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "name": config.wandb.run_name or config.output.run_name,
        "tags": list(config.wandb.tags),
        "config": config.model_dump(by_alias=True, mode="json"),
        "dir": str(run_dir),
        "mode": mode,
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity

    logger = None
    try:
        logger = setup_wandb_logging(**logging_kwargs, **init_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external service
        if mode != "offline":
            logging.getLogger("trainer").warning("W&B initialization failed (%s); retrying in offline mode.", exc)
            try:
                init_kwargs["mode"] = "offline"
                logger = setup_wandb_logging(**logging_kwargs, **init_kwargs)
            except Exception as offline_exc:  # pragma: no cover
                logging.getLogger("trainer").warning(
                    "W&B offline initialization failed (%s); W&B logging disabled.", offline_exc
                )
                return None
        else:
            logging.getLogger("trainer").warning("W&B initialization failed (%s); W&B logging disabled.", exc)
            return None

    try:
        (run_dir / "wandb_run_id.txt").write_text(str(logger.run.id))
    except Exception:  # pragma: no cover - best effort only
        pass
    return logger


def write_eval_context(run_dir: Path, **values: Any) -> None:
    path = Path(run_dir) / "eval_context.json"
    path.write_text(json.dumps(values, indent=2, sort_keys=True))


def read_eval_context(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "eval_context.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
