"""Pack saved raw NCA grids with the published 2×2 tokenizer."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class PackingConfig:
    num_colors: int
    patch_size: int = 2

    def __post_init__(self):
        if self.num_colors < 2 or self.patch_size < 1:
            raise ValueError("num_colors must be at least 2 and patch_size positive")
        if self.num_colors ** (self.patch_size ** 2) > 2**31 - 2:
            raise ValueError("patch vocabulary exceeds the original int32 token range")


def pack_one(states: np.ndarray, *, num_colors: int, patch_size: int = 2):
    """Pack one world with the upstream patch order and grid markers."""
    states = np.asarray(states)
    if states.ndim != 4 or states.shape[-1] != 1:
        raise ValueError("legacy tokenizer requires [frames, height, width, 1]")
    if states.shape[1] % patch_size or states.shape[2] % patch_size:
        raise ValueError("grid dimensions must be divisible by patch_size")
    grid = jnp.asarray(states[None], dtype=jnp.int32)
    batch, frames, height, width, _ = grid.shape
    patch_rows = height // patch_size
    patch_columns = width // patch_size
    grid = grid.reshape(batch, frames, height, width)
    grid = grid.reshape(batch, frames, patch_rows, patch_size,
                        patch_columns, patch_size)
    grid = grid.transpose(0, 1, 2, 4, 3, 5)
    grid = grid.reshape(batch, frames, patch_rows * patch_columns,
                        patch_size * patch_size)
    powers = num_colors ** jnp.arange(patch_size * patch_size)
    patches = jnp.einsum("bnlp,p->bnl", grid, powers)
    mask = jnp.full((batch, frames, 1), -100, dtype=patches.dtype)
    start = jnp.full((batch, frames, 1), num_colors ** (patch_size ** 2),
                     dtype=patches.dtype)
    end = jnp.full((batch, frames, 1), num_colors ** (patch_size ** 2) + 1,
                   dtype=patches.dtype)
    sequence = jnp.concat([start, patches, end], axis=-1).reshape(batch, -1)
    targets = jnp.concat([mask, patches, mask], axis=-1).reshape(batch, -1)
    return torch.tensor(sequence[0]), torch.tensor(targets[0])


@dataclass
class PackedDataset(Dataset):
    sequences: torch.Tensor  # [trajectories, tokens]
    targets: torch.Tensor  # original tokenizer's marker-masked targets
    metadata: dict

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sequences": self.sequences, "targets": self.targets,
                    "metadata": self.metadata}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PackedDataset":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(payload["sequences"], payload["targets"], payload["metadata"])

    def split(self, validation_count: int) -> tuple["PackedDataset", "PackedDataset"]:
        if not 0 < validation_count < len(self):
            raise ValueError("validation_count must be between 1 and dataset length - 1")
        n = len(self) - validation_count
        return (PackedDataset(self.sequences[:n], self.targets[:n], self.metadata),
                PackedDataset(self.sequences[n:], self.targets[n:], self.metadata))

    def training_pairs(self, context_length: int, min_grid: int = 1) -> Dataset:
        """Use the token branch of the legacy NCADataset masking and shift.

        The published NCADataset takes its next-token targets from `seq`, masks
        the first `min_grid` grids, then shifts by one. Its `targets` argument is
        not used in this branch. Requiring a full context avoids its malformed
        padding branch while preserving the tokens and loss positions exactly.
        """
        grid_len = self.metadata["grid_len"]
        if context_length < 1 or context_length > self.sequences.shape[1] - 1:
            raise ValueError("context_length must fit the packed trajectory")
        target = self.sequences.clone().long()
        target[target < 0] = -100
        target[:, :min_grid * grid_len] = -100
        inputs = self.sequences[:, :context_length].long()
        labels = target[:, 1:context_length + 1]
        return torch.utils.data.TensorDataset(inputs, labels)


def pack_dataset(raw, config: PackingConfig | None = None) -> PackedDataset:
    """Pack a saved nca.TrajectoryDataset without changing its files."""
    generator_config = raw.metadata["config"]
    config = config or PackingConfig(generator_config["num_colors"],
                                     generator_config["patch_size"])
    if config.num_colors < generator_config["num_colors"]:
        raise ValueError("packing alphabet cannot be smaller than generated alphabet")
    sequences, targets = [], []
    for record in raw:
        seq, target = pack_one(record["states"], num_colors=config.num_colors,
                               patch_size=config.patch_size)
        sequences.append(seq)
        targets.append(target)
    if not sequences:
        raise ValueError("cannot pack an empty trajectory selection")
    grid_len = (generator_config["grid_size"] // config.patch_size) ** 2 + 2
    metadata = {"source_manifest": raw.metadata, "grid_len": grid_len,
                "num_colors": config.num_colors, "patch_size": config.patch_size,
                "packing_config": asdict(config),
                "source_directory": str(raw.directory.resolve()),
                "source_manifest_sha256": hashlib.sha256(
                    (raw.directory / "manifest.json").read_bytes()).hexdigest(),
                "trajectory_indices": [record["index"] for record in raw]}
    return PackedDataset(torch.stack(sequences), torch.stack(targets), metadata)
