"""Read raw NCA datasets without importing the simulator."""

import json
from pathlib import Path
from collections.abc import Callable, Iterator
import shutil

import numpy as np


def read_manifest(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text())


def iter_trajectories(directory: str | Path):
    """Yield saved trajectory records; NumPy maps each shard instead of copying it."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    rules = np.load(directory / manifest["rules"], mmap_mode="r")
    index = 0
    for shard in manifest["shards"]:
        arrays = {
            name: np.load(directory / shard[name], mmap_mode="r")
            for name in ("states", "rule_ids", "initial_seeds", "initial_states")
        }
        for row in range(shard["count"]):
            rule_id = int(arrays["rule_ids"][row])
            yield {
                "index": index,
                "states": arrays["states"][row],
                "initial_state": arrays["initial_states"][row],
                "rule_id": rule_id,
                "rule_seed": rules[rule_id],
                "initial_seed": arrays["initial_seeds"][row],
            }
            index += 1


def get_trajectory(directory: str | Path, index: int) -> dict:
    if index < 0:
        raise IndexError("trajectory index must be nonnegative")
    for record in iter_trajectories(directory):
        if record["index"] == index:
            return record
    raise IndexError(f"trajectory {index} does not exist")


class TrajectoryDataset:
    """A small read-only view of a saved version 1 trajectory dataset.

    Records remain memory-mapped views of the original .npy shards. Filtering
    selects records in memory and never changes the disk schema or file order.
    """

    def __init__(self, directory: str | Path, indices: tuple[int, ...] | None = None):
        self.directory = Path(directory)
        self.metadata = read_manifest(self.directory)
        if self.metadata.get("schema_version") != 1:
            raise ValueError("expected NCA trajectory schema version 1")
        self._total = sum(shard["count"] for shard in self.metadata["shards"])
        self._indices = indices

    @classmethod
    def load(cls, directory: str | Path) -> "TrajectoryDataset":
        return cls(directory)

    def __len__(self) -> int:
        return self._total if self._indices is None else len(self._indices)

    def __iter__(self) -> Iterator[dict]:
        selected = None if self._indices is None else set(self._indices)
        for record in iter_trajectories(self.directory):
            if selected is None or record["index"] in selected:
                yield record

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        disk_index = index if self._indices is None else self._indices[index]
        return get_trajectory(self.directory, disk_index)

    def filter(self, predicate: Callable[[dict], bool]) -> "TrajectoryDataset":
        """Select records by metadata or computed statistics, preserving order."""
        indices = tuple(record["index"] for record in self if predicate(record))
        return TrajectoryDataset(self.directory, indices)

    def save(self, output: str | Path) -> Path:
        """Copy an entire v1 dataset byte-for-byte to a new directory."""
        if self._indices is not None:
            raise ValueError("filtered views cannot be saved as a full v1 dataset")
        return Path(shutil.copytree(self.directory, output))
