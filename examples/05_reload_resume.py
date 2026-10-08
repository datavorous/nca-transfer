"""Resume the tiny run for ten more optimizer steps on the same packed data."""

from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_nca_pretraining.training import (TrainingConfig, load_checkpoint,
                                                train, save_checkpoint)


packed = PackedDataset.load("runs/tiny/packed.pt")
training, validation = packed.split(validation_count=1)
checkpoint = "runs/tiny/checkpoints/model_20_iter.pth"
config = TrainingConfig(steps=30, batch_size=2, learning_rate=1e-3,
                        eval_interval=5, seed=7)
state = load_checkpoint(checkpoint, training_config=config)
context = state.model_config.context_length
train(state, training.training_pairs(context), validation.training_pairs(context))
print(save_checkpoint(state, "runs/tiny/checkpoints"))
