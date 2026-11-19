import html
import logging
import random
from pathlib import Path

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver
from ignite.handlers.clearml_logger import ClearMLSaver
from ignite.handlers.logger_utils import setup_clearml_logging, setup_tb_logging
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import setup_logger
from torch.utils.data import DataLoader

from config import (
    ProjectConfig,
    TrainerConfig,
)
from evaluation import evaluate_exact_match, evaluate_split

from utils import (
    attach_early_stopping,
    attach_lr_scheduler,
    build_dataloaders,
    build_model,
    create_run_dir,
    instantiate,
    setup_best_model_checkpoint,
    setup_console_logger,
    setup_training_checkpointing,
    setup_wandb_logging_safe,
    write_eval_context,
)


def _flatten(output: dict[str, torch.Tensor], ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    logits = output["logits"].reshape(-1, output["logits"].size(-1))
    targets = output["targets"].reshape(-1)
    # Build a boolean mask that is True for every non-ignored position
    mask = targets != ignore_index
    return logits[mask], targets[mask]


class Trainer:

    def __init__(self, config: ProjectConfig) -> None:
        self.logger: logging.Logger = setup_logger("trainer", level=logging.INFO)
        # Fix randomness for reproducibility across runs with the same seed
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.config: ProjectConfig = config
        self.trainer_config: TrainerConfig = config.trainer

        # Device placement via Ignite's distributed helper
        self.device = idist.device() if config.device is None else config.device

        # Build tokenizer from config (e.g. CharTokenizer)
        self.tokenizer = instantiate(config.preprocessor)
        # Build train/val/test DataLoaders from the data config
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            config.data,
            self.tokenizer,
            config.seed,
            collate_fn_target=config.model.collate_fn_target,
        )
        self.model = build_model(
            config.model, self.device, vocab_size=self.tokenizer.vocab_size, pad_token_id=self.tokenizer.pad_token_id
        )
        self.optimizer = instantiate(config.optimizer, params=self.model.parameters())
        self.epochs: int = self.trainer_config.epochs
        self.grad_clip_norm: float | None = getattr(self.trainer_config, "grad_clip_norm", None)
        self.inference_every_epochs = self.trainer_config.inference_every_epochs
        self.inference_num_samples = self.trainer_config.inference_num_samples
        self.max_inference_new_tokens = self.trainer_config.max_inference_new_tokens
        self.criterion = instantiate(config.loss)
        self.ignore_index: int = getattr(self.criterion, "ignore_index", -100)
        self.requires_router_info = getattr(self.criterion, "requires_router_info", False)
        self.train_forward_kwargs = {"return_router_info": True} if self.requires_router_info else {}


        self.total_iters: int = len(self.train_loader) * self.epochs
        self.trainer = Engine(self._train_step)
        self.train_evaluator = Engine(self._eval_step)
        self.val_evaluator = Engine(self._eval_step)
        self.test_evaluator = Engine(self._eval_step)
        self.metrics: dict = self._attach_metrics()

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_iteration = 0
        self.val_evaluator.add_event_handler(Events.COMPLETED, self._track_best)

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._run_evaluations)

        # Create run directory on rank 0 and broadcast to all ranks
        self.run_dir = create_run_dir(config)

        # Attach loggers for console
        self.logger = setup_console_logger(
            self.run_dir, self.trainer, self.trainer_config.log_every_steps, self.optimizer
        )

        # Attach learning rate scheduler
        attach_lr_scheduler(self.trainer, self.optimizer, self.total_iters, config.optimizer.lr)

        # Attach early stopping
        if self.trainer_config.early_stopping_patience is not None:
            attach_early_stopping(self.val_evaluator, self.trainer, self.trainer_config.early_stopping_patience)

        # Define validation scoring function for best model
        def score_function(engine_val):
            loss = engine_val.state.metrics["loss"]
            return -loss

        evaluators = {"training": self.train_evaluator, "validation": self.val_evaluator}
        logging_kwargs = {
            "trainer": self.trainer,
            "optimizers": None,
            "evaluators": evaluators,
            "log_every_iters": self.trainer_config.log_every_steps,
            "trainer_metric_names": ["batch_loss"],
            "evaluator_metric_names": ["loss", "token_accuracy"],
        }
        self.exp_logger = None
        self.wandb_logger = None
        if idist.get_rank() == 0:
            if config.logger == "clearml":
                self.exp_logger = setup_clearml_logging(
                    **logging_kwargs,
                    project_name=config.project_name,
                    task_name=self.run_dir.name,
                )
                self.exp_logger.get_task().upload_artifact(
                    name="config.yaml", artifact_object=str(self.run_dir / "config.yaml")
                )
            else:
                self.exp_logger = setup_tb_logging(
                    output_path=str(self.run_dir / "tensorboard"),
                    **logging_kwargs,
                )
            # W&B is additive: TensorBoard remains the local source of truth.
            self.wandb_logger = setup_wandb_logging_safe(config, self.run_dir, logging_kwargs)

        # Progress scalars, learning rate, and optional periodic exact match.
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch_progress)
        if self.trainer_config.exact_match_every_epochs is not None:
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.trainer_config.exact_match_every_epochs),
                self._run_periodic_id_eval,
            )

        # Setup save handler and checkpointing on main process only
        if idist.get_rank() == 0:
            if config.logger == "clearml":
                save_handler = ClearMLSaver(
                    logger=self.exp_logger,
                    dirname=str(self.run_dir),
                    output_uri=True,
                    require_empty=False,
                )
            else:
                save_handler = DiskSaver(dirname=str(self.run_dir), require_empty=False)

            unwrapped_model: torch.nn.Module = self.model
            self.best_checkpoint = setup_best_model_checkpoint(
                self.trainer,
                self.val_evaluator,
                {"model": unwrapped_model, "optimizer": self.optimizer},
                save_handler,
                score_function=score_function,
                score_name="val_loss",
            )
            self.last_checkpoint = setup_training_checkpointing(
                self.trainer,
                {"model": unwrapped_model, "optimizer": self.optimizer},
                save_handler,
            )
        else:
            self.best_checkpoint = None
            self.last_checkpoint = None
        # Attach inference logger if inference logging is enabled
        if self.inference_every_epochs is not None:
            self.attach_inference_logger()

    def _track_best(self, engine: Engine) -> None:
        loss = float(engine.state.metrics["loss"])
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_epoch = self.trainer.state.epoch
            self.best_iteration = self.trainer.state.iteration

    def _log_scalars(self, metrics: dict[str, float], step: int) -> None:
        clean = {key: float(value) for key, value in metrics.items() if value is not None}
        if not clean:
            return
        if self.exp_logger is not None:
            if self.config.logger == "clearml":
                for key, value in clean.items():
                    self.exp_logger.report_scalar(title=key, series="value", value=value, iteration=step)
            else:
                for key, value in clean.items():
                    self.exp_logger.writer.add_scalar(key, value, step)
        if self.wandb_logger is not None:
            try:
                self.wandb_logger.log(clean, step=step)
            except Exception as exc:  # pragma: no cover - external service
                self.logger.warning(
                    "W&B logging with explicit step failed (%s); retrying without step.", exc
                )
                try:
                    self.wandb_logger.log(clean)
                except Exception as retry_exc:  # pragma: no cover - external service
                    self.logger.warning("W&B logging failed (%s); continuing without W&B.", retry_exc)
                    self.wandb_logger = None

    def _log_epoch_progress(self, engine: Engine) -> None:
        step = engine.state.iteration
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self._log_scalars(
            {
                "training/learning_rate": learning_rate,
                "progress/epoch": float(engine.state.epoch),
                "progress/global_step": float(step),
            },
            step=step,
        )

    def _run_periodic_id_eval(self, engine: Engine) -> None:
        if idist.get_rank() != 0:
            return
        unwrapped_model: torch.nn.Module = self.model
        result = evaluate_exact_match(
            unwrapped_model,
            self.tokenizer,
            self.config.data.val,
            self.device,
            max_new_tokens=self.max_inference_new_tokens,
            batch_size=self.trainer_config.eval_batch_size,
        )
        step = engine.state.iteration
        metrics = {
            "eval/id/exact_match": result.exact_match,
            "eval/id/num_samples": float(result.num_samples),
            "progress/epoch": float(engine.state.epoch),
            "progress/global_step": float(step),
        }
        val_token_accuracy = self.val_evaluator.state.metrics.get("token_accuracy")
        if val_token_accuracy is not None:
            metrics["eval/id/token_accuracy"] = float(val_token_accuracy)
        for key, stats in result.per_length.items():
            metrics[f"eval/id/exact_match/{key}"] = stats["accuracy"]
        self._log_scalars(metrics, step=step)
        self.logger.info(
            "epoch=%s ID exact_match=%.4f (%d samples)",
            engine.state.epoch,
            result.exact_match,
            result.num_samples,
        )

    def _train_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.train()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs, attention_mask=attention_mask, **self.train_forward_kwargs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        loss = self.criterion(outputs, targets)

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "logits": logits.detach(),
            "targets": targets.detach(),
        }

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.eval()
        inputs = batch["input_ids"].to(self.device)
        targets = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(inputs, attention_mask=attention_mask)

        if idist.get_world_size() > 1:
            logits = logits
            targets = targets

        # Gather across all ranks and drop duplicated padding samples on the last batch
        logits, targets = (logits, targets)
        return {"logits": logits, "targets": targets}

    def _run_evaluations(self, engine: Engine) -> None:
        self.logger.info("Evaluating on training set...")
        self.train_evaluator.run(self.train_loader, epoch_length=min(len(self.train_loader), len(self.val_loader)))
        train_metrics = self.train_evaluator.state.metrics
        epoch = engine.state.epoch

        self.logger.info("Evaluating on validation set...")
        self.val_evaluator.run(self.val_loader)
        val_metrics = self.val_evaluator.state.metrics
        self.logger.info(
            "epoch=%s train_loss=%.4f train_token_acc=%.4f val_loss=%.4f val_token_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
        )

    def run(self) -> None:
        self.logger.info("starting run in %s", self.run_dir)
        try:
            self.trainer.run(self.train_loader, max_epochs=self.epochs)
            self.test()
        finally:
            if self.exp_logger is not None:
                self.exp_logger.close()
            if self.wandb_logger is not None:
                try:
                    self.wandb_logger.close()
                except Exception:  # pragma: no cover - external service
                    pass

    def test(self, test_loader: DataLoader | None = None) -> None:
        idist.barrier()
        if idist.get_rank() != 0:
            return

        checkpoint_handler = self.best_checkpoint
        checkpoint_path = None
        if checkpoint_handler and checkpoint_handler.last_checkpoint:
            checkpoint_path = Path(checkpoint_handler.last_checkpoint)
        elif (self.run_dir / "best.pt").exists():
            checkpoint_path = self.run_dir / "best.pt"

        unwrapped_model: torch.nn.Module = self.model
        if checkpoint_path is not None:
            self.logger.info("Loading best model for final evaluation from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            unwrapped_model.load_state_dict(checkpoint["model"])
        else:
            self.logger.warning("No best model checkpoint found. Using current model for final evaluation.")

        step = self.best_iteration or self.trainer.state.iteration
        epoch = self.best_epoch or self.trainer.state.epoch
        collate_fn_target = self.config.model.collate_fn_target or "models.rope_transformer.CausalLMCollateFn"

        metrics: dict[str, float] = {
            "progress/epoch": float(epoch),
            "progress/global_step": float(step),
            "eval/best_val_loss": float(self.best_val_loss) if self.best_val_loss != float("inf") else float("nan"),
        }

        id_result, id_token_accuracy = evaluate_split(
            unwrapped_model,
            self.tokenizer,
            self.config.data.val,
            self.device,
            max_new_tokens=self.max_inference_new_tokens,
            collate_fn_target=collate_fn_target,
            batch_size=self.trainer_config.eval_batch_size,
        )
        metrics.update(
            {
                "eval/id/exact_match": id_result.exact_match,
                "eval/id/token_accuracy": id_token_accuracy,
                "eval/id/num_samples": float(id_result.num_samples),
            }
        )
        for key, stats in id_result.per_length.items():
            metrics[f"eval/id/exact_match/{key}"] = stats["accuracy"]
        self.logger.info(
            "ID (validation) exact_match=%.4f token_accuracy=%.4f (%d samples)",
            id_result.exact_match,
            id_token_accuracy,
            id_result.num_samples,
        )
        for source, target, prediction in id_result.wrong:
            self.logger.info("  ID wrong: %s = %s -> %s", source, target, prediction)

        test_split = getattr(self.config.data, "test", None)
        if test_split is not None:
            ood_result, ood_token_accuracy = evaluate_split(
                unwrapped_model,
                self.tokenizer,
                test_split,
                self.device,
                max_new_tokens=self.max_inference_new_tokens,
                collate_fn_target=collate_fn_target,
                batch_size=self.trainer_config.eval_batch_size,
            )
            metrics.update(
                {
                    "eval/ood/exact_match": ood_result.exact_match,
                    "eval/ood/token_accuracy": ood_token_accuracy,
                    "eval/ood/num_samples": float(ood_result.num_samples),
                }
            )
            for key, stats in ood_result.per_length.items():
                metrics[f"eval/ood/exact_match/{key}"] = stats["accuracy"]
            self.logger.info(
                "OOD (test) exact_match=%.4f token_accuracy=%.4f (%d samples)",
                ood_result.exact_match,
                ood_token_accuracy,
                ood_result.num_samples,
            )
            for source, target, prediction in ood_result.wrong:
                self.logger.info("  OOD wrong: %s = %s -> %s", source, target, prediction)
        else:
            self.logger.warning("No OOD test split configured; skipping OOD evaluation.")

        self._log_scalars(metrics, step=step)
        write_eval_context(
            self.run_dir,
            best_epoch=epoch,
            best_iteration=step,
            last_epoch=self.trainer.state.epoch,
            last_iteration=self.trainer.state.iteration,
            best_val_loss=self.best_val_loss,
            metrics=metrics,
        )

    def _attach_metrics(self) -> dict[str, Metric]:
        # Running average loss tracked per training iteration (logged in console)
        RunningAverage(output_transform=lambda output: output["loss"]).attach(self.trainer, "batch_loss")

        # Shared transform: flatten and filter out ignored positions for both metrics
        def transform_fn(output):
            return _flatten(output, ignore_index=self.ignore_index)

        eval_criterion = self.criterion.cross_entropy if self.requires_router_info else self.criterion
        metrics = {}
        for prefix, evaluator in [
            ("train", self.train_evaluator),
            ("val", self.val_evaluator),
            ("test", self.test_evaluator),
        ]:
            loss = Loss(eval_criterion, output_transform=transform_fn, device=self.device)
            token_acc = Accuracy(output_transform=transform_fn, device=self.device)

            loss.attach(evaluator, "loss")
            token_acc.attach(evaluator, "token_accuracy")

            metrics[f"{prefix}_loss"] = loss
            metrics[f"{prefix}_token_accuracy"] = token_acc

        return metrics

    def attach_inference_logger(self) -> None:
        for loader, name in ((self.train_loader, "Train"), (self.val_loader, "Val")):
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=self.inference_every_epochs),
                self._log_inference,
                loader,
                name,
            )

    def _log_text(self, tag: str, text: str, step: int) -> None:
        if self.exp_logger is None:
            return
        if self.config.logger == "clearml":
            self.exp_logger.report_text(f"[{tag}] Step {step}:\n{text}")
        else:
            self.exp_logger.writer.add_text(tag, text, global_step=step)

    def _log_inference(self, engine: Engine, loader: DataLoader, name: str) -> None:
        if idist.get_rank() != 0:
            return
        self.logger.info(f"Epoch {engine.state.epoch}: Running inference on {name} samples...")

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        dataset = loader.dataset
        total_samples = len(dataset)  # type: ignore
        num_samples = min(self.inference_num_samples, total_samples)

        prompt_ids_list: list[torch.Tensor] = []
        targets: list[str] = []
        sources: list[str] = []
        reverse_for_display: list[bool] = []
        for i in range(num_samples):
            item = dataset[i]
            lsd_first = getattr(item, "digit_order", None) == "lsd_first"
            source = item.original_source if getattr(item, "original_source", None) is not None else item.source
            target = item.original_target if getattr(item, "original_target", None) is not None else item.target

            input_ids = item.eval_input_ids
            prompt_ids_list.append(input_ids)
            targets.append(target)
            sources.append(source)
            reverse_for_display.append(lsd_first)
        self.model.eval()

        batch_input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                [t.flip(0) for t in prompt_ids_list], batch_first=True, padding_value=pad_token_id
            )
            .flip(1)
            .to(self.device)
        )
        batch_attention_mask = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(t, dtype=torch.long).flip(0) for t in prompt_ids_list],
                batch_first=True,
                padding_value=0,
            )
            .flip(1)
            .to(self.device)
        )
        sequences = self.generate(
            input_ids=batch_input_ids,
            max_new_tokens=self.max_inference_new_tokens,
            attention_mask=batch_attention_mask,
        )
        new_tokens = sequences[:, batch_input_ids.shape[1] :]
        decoded_strs = [
            self.tokenizer.decode(
                tokens.tolist(),
                skip_special_tokens=True,
            )
            for tokens in new_tokens
        ]
        decoded_strs = [
            prediction[::-1] if reverse else prediction
            for prediction, reverse in zip(decoded_strs, reverse_for_display)
        ]

        for idx, (source, target, prediction) in enumerate(zip(sources, targets, decoded_strs), 1):
            self.logger.info(
                "    Sample %d | Prompt: %r | Target: %r | Prediction: %r",
                idx,
                source,
                target,
                prediction,
            )

        lines = []
        for idx, (source, target, prediction) in enumerate(zip(sources, targets, decoded_strs), 1):
            lines.append(f"Sample {idx}")
            for key, val in [("Prompt", source), ("Target", target), ("Prediction", prediction)]:
                val = html.escape(val.strip() or "(empty)", quote=False)
                lines.append(f"  {key}:     {val}")
            lines.append("")
        name_map = {"Train": "training", "Val": "validation", "Test": "testing"}
        tag = f"inference/{name_map.get(name, name.lower())}"
        self._log_text(tag, "\n".join(lines), engine.state.iteration)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        unwrapped_model = self.model
        unwrapped_model.eval()

        eos_id = self.tokenizer.eos_token_id

        device = self.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = unwrapped_model(generated, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            # Greedy decoding: pick the highest-probability token at each step
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if eos_id is not None:
                eos_mask = generated[:, -1:].eq(eos_id)
                next_token = torch.where(eos_mask, torch.tensor(eos_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=-1)

            # Extend the attention mask by one column for the newly appended token
            next_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=device,
            )

            already_ended = generated[:, -2:-1].eq(eos_id)
            next_mask = torch.where(
                already_ended,
                torch.tensor(0, dtype=attention_mask.dtype, device=device),
                next_mask,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            # Stop early if every sequence in the batch has produced EOS
            if generated[:, -1].eq(eos_id).all():
                break

        return generated
