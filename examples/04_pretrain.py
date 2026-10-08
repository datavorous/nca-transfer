"""Train a tiny copy of the original model on already packed NCA data."""

from sampleeff_nca_pretraining.model import ModelConfig, create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_nca_pretraining.training import (TrainingConfig, prepare, train,
                                                save_checkpoint, export_results)


packed = PackedDataset.load("runs/tiny/packed.pt")
training, validation = packed.split(validation_count=1)
context = packed.sequences.shape[1] - 1
model_config = ModelConfig(vocab_size=packed.metadata["num_colors"] **
                           packed.metadata["patch_size"] ** 2 + 2,
                           context_length=context, layers=1, heads=2,
                           embedding_dim=32)
train_config = TrainingConfig(steps=20, batch_size=2, learning_rate=1e-3,
                              eval_interval=5, seed=7)
model = create_model(model_config, seed=11)
print("parameters:", trainable_parameters(model))
state = prepare(model, model_config, train_config, dataset_metadata=packed.metadata,
                model_seed=11)
train(state, training.training_pairs(context), validation.training_pairs(context))
checkpoint = save_checkpoint(state, "runs/tiny/checkpoints")
export_results(state, "runs/tiny/results", checkpoint=checkpoint)
print(checkpoint)
