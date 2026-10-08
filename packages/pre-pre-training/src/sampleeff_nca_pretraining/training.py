"""Small Python training loop for prepacked NCA tokens.

This keeps the published model, causal mask, Adam optimizer, scheduler and
checkpoint keys. It accepts saved data instead of generating NCA rules inside
the training loop.
"""

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from sampleeff_nca_pretraining.model import ModelConfig, create_model
from sampleeff_shared.models import create_attention_mask
from sampleeff_shared.runtime import get_lr_scheduler, save_checkpoint as save_legacy_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    eval_interval: int = 5
    checkpoint_interval: int | None = None
    device: str = "auto"
    seed: int = 0

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.eval_interval) < 1:
            raise ValueError("steps, batch_size and eval_interval must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("invalid optimizer settings")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")


@dataclass
class TrainingState:
    model: torch.nn.Module
    model_config: ModelConfig
    config: TrainingConfig
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device
    step: int = 0
    history: list[dict] = field(default_factory=list)
    dataset_metadata: dict | None = None
    model_seed: int | None = None


def _device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def prepare(model: torch.nn.Module, model_config: ModelConfig,
            config: TrainingConfig, *, dataset_metadata: dict | None = None,
            model_seed: int | None = None) -> TrainingState:
    """Attach the original optimizer/scheduler family to an existing model."""
    torch.manual_seed(config.seed)
    device = _device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.steps)
    return TrainingState(model, model_config, config, optimizer, scheduler, device,
                         dataset_metadata=dataset_metadata, model_seed=model_seed)


def _loss(state: TrainingState, batch, criterion, mask):
    inputs, targets = (tensor.to(state.device) for tensor in batch)
    if inputs.shape[1] != state.model_config.context_length:
        raise ValueError("packed input length must equal model context_length")
    logits = state.model(inputs, attention_mask=mask)
    return criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def evaluate(state: TrainingState, dataset) -> float:
    state.model.eval()
    criterion = CrossEntropyLoss()
    mask = create_attention_mask(state.model_config.context_length, additive=True).to(state.device)
    losses = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=state.config.batch_size):
            losses.append(float(_loss(state, batch, criterion, mask)))
    if not losses:
        raise ValueError("validation dataset is empty")
    return sum(losses) / len(losses)


def train(state: TrainingState, training_data, validation_data=None,
          *, checkpoint_dir: str | Path | None = None) -> TrainingState:
    """Train to `config.steps`; call again after loading a checkpoint to resume."""
    if not len(training_data):
        raise ValueError("training dataset is empty")
    if state.config.checkpoint_interval is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint_interval is set")
    generator = torch.Generator().manual_seed(state.config.seed + state.step)
    loader = DataLoader(training_data, batch_size=state.config.batch_size,
                        shuffle=True, generator=generator)
    batches = iter(loader)
    criterion = CrossEntropyLoss()
    mask = create_attention_mask(state.model_config.context_length, additive=True).to(state.device)
    for step in range(state.step + 1, state.config.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        loss = _loss(state, batch, criterion, mask)
        loss.backward()
        state.optimizer.step()
        state.scheduler.step()
        state.step = step
        row = {"step": step, "train_loss": float(loss.detach().cpu()),
               "learning_rate": state.optimizer.param_groups[0]["lr"],
               "validation_loss": None}
        if validation_data is not None and (step % state.config.eval_interval == 0 or step == state.config.steps):
            row["validation_loss"] = evaluate(state, validation_data)
        state.history.append(row)
        if (state.config.checkpoint_interval is not None
                and step % state.config.checkpoint_interval == 0):
            save_checkpoint(state, checkpoint_dir)
    return state


def save_checkpoint(state: TrainingState, directory: str | Path) -> Path:
    """Write the legacy .pth keys plus a JSON sidecar for model recreation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validation = [row["validation_loss"] for row in state.history
                  if row["validation_loss"] is not None]
    best = min(validation, default=float("inf"))
    save_legacy_checkpoint(
        epoch=0, count=state.step, model=state.model, optimizer=state.optimizer,
        scheduler=state.scheduler, best_val_loss=best, best_val_loss_el=best,
        metrics={"history": state.history}, save_dir=str(directory),
        total_iterations=state.step, by_epoch=False,
    )
    checkpoint = directory / f"model_{state.step}_iter.pth"
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(json.dumps({"model_config": asdict(state.model_config),
                                   "training_config": asdict(state.config),
                                   "dataset_metadata": state.dataset_metadata,
                                   "model_seed": state.model_seed}, indent=2) + "\n")
    return checkpoint


def load_checkpoint(path: str | Path, model_config: ModelConfig | None = None,
                    training_config: TrainingConfig | None = None) -> TrainingState:
    """Load a new checkpoint, or an original .pth with explicit configurations."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
        model_config = model_config or ModelConfig(**saved["model_config"])
        training_config = training_config or TrainingConfig(**saved["training_config"])
    if model_config is None or training_config is None:
        raise ValueError("legacy checkpoints need model_config and training_config")
    state = prepare(create_model(model_config), model_config, training_config)
    payload = torch.load(path, map_location=state.device, weights_only=False)
    state.model.load_state_dict(payload["model"])
    if "optimizer" in payload:
        state.optimizer.load_state_dict(payload["optimizer"])
    if "scheduler" in payload:
        state.scheduler.load_state_dict(payload["scheduler"])
    state.step = payload.get("total_iterations", payload.get("count", 0))
    state.history = payload.get("metrics", {}).get("history", [])
    state.dataset_metadata = saved.get("dataset_metadata") if sidecar.exists() else None
    state.model_seed = saved.get("model_seed") if sidecar.exists() else None
    return state


def load_pretrained_body(model: torch.nn.Module, path: str | Path) -> None:
    """Transfer transformer body while allowing a new vocabulary/head."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    body = {key: value for key, value in source.items()
            if not key.startswith(("input_proj.", "output_proj."))}
    incompatible = model.load_state_dict({**target, **body}, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible transformer body: {incompatible}")


def export_results(state: TrainingState, directory: str | Path,
                   *, checkpoint: str | Path | None = None,
                   dataset_metadata: dict | None = None) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"model_config": asdict(state.model_config),
               "training_config": asdict(state.config),
               "model_seed": state.model_seed,
               "step": state.step, "checkpoint": str(checkpoint) if checkpoint else None,
               "final_validation_loss": next((row["validation_loss"] for row in reversed(state.history)
                                              if row["validation_loss"] is not None), None),
               "dataset_metadata": dataset_metadata if dataset_metadata is not None
               else state.dataset_metadata}
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    history_path = directory / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "train_loss", "validation_loss", "learning_rate"))
        writer.writeheader()
        writer.writerows(state.history)
    return summary_path, history_path
