"""Tiny NCA-to-checkpoint smoke reproduction with an intentional overfit check.

Run with `uv run --package sampleeff-nca-pre-pretraining python examples/reproduce_tiny.py OUTPUT_DIR`.
The validation set is the training set here so decreasing loss checks the
machinery, not generalization to unseen rules.
"""

import sys
from pathlib import Path

import torch

from nca import GameConfig, NCAGenerator, TrajectoryDataset
from nca.analyze import gzip_ratio
from nca.visualize import plot
from sampleeff_nca_pretraining.model import ModelConfig, create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset, pack_dataset
from sampleeff_nca_pretraining.training import (TrainingConfig, evaluate, export_results,
                                                load_checkpoint, prepare, save_checkpoint, train)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/tiny-reproduction")
    game = GameConfig(grid_size=4, num_colors=3, rule_count=3,
                      trajectories_per_rule=1, frames=4, burn_in=1,
                      batch_size=2, patch_size=2, seed=7)
    NCAGenerator(game).save(root / "raw")
    raw = TrajectoryDataset.load(root / "raw")
    assert len(raw) == 3
    print("gzip ratio:", gzip_ratio(raw[0]["states"], game.num_colors))
    plot(raw.directory, 0, root / "world.png")

    packed = PackedDataset.load(pack_dataset(raw).save(root / "packed.pt"))
    context = packed.sequences.shape[1] - 1
    pairs = packed.training_pairs(context, min_grid=1)
    model_config = ModelConfig(vocab_size=game.num_colors ** (game.patch_size ** 2) + 2,
                               context_length=context, layers=1, heads=2,
                               embedding_dim=32, dropout=0.0)
    train_config = TrainingConfig(steps=30, batch_size=3, learning_rate=2e-3,
                                  eval_interval=10, device="cpu", seed=13)
    model = create_model(model_config, seed=11)
    print("trainable parameters:", trainable_parameters(model))
    state = prepare(model, model_config, train_config, dataset_metadata=packed.metadata,
                    model_seed=11)
    first_loss = evaluate(state, pairs)
    train(state, pairs, pairs)
    final_loss = evaluate(state, pairs)
    print("validation loss (same worlds):", first_loss, "->", final_loss)
    assert final_loss < first_loss

    checkpoint = save_checkpoint(state, root / "checkpoints")
    restored = load_checkpoint(checkpoint)
    state.model.eval()
    restored.model.eval()
    tokens = pairs[0][0][None]
    with torch.no_grad():
        torch.testing.assert_close(state.model(tokens), restored.model(tokens))
    export_results(restored, root / "results", checkpoint=checkpoint)
    print("checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
