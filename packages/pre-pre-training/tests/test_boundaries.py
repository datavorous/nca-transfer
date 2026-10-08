"""Boundary checks against tiny outputs frozen from upstream revision bdd1c71."""

from dataclasses import asdict
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from nca.analyze import gzip_ratio
from nca.config import GameConfig
from nca.generate import reproduce
from sampleeff_nca_pretraining.model import ModelConfig, create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one
from sampleeff_nca_pretraining.training import (TrainingConfig, load_checkpoint,
                                                load_pretrained_body, prepare,
                                                save_checkpoint, train)


FIXTURE = json.loads((Path(__file__).parent / "fixtures/upstream_tiny.json").read_text())


class BoundaryTest(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_small_model_forward_on_mps(self):
        model = create_model(ModelConfig(vocab_size=83, context_length=5,
                                         layers=1, heads=2, embedding_dim=32), seed=1).to("mps")
        with torch.no_grad():
            result = model(torch.zeros((1, 5), dtype=torch.long, device="mps"))
        self.assertEqual(result.shape, (1, 5, 83))

    def test_fixed_seeds_match_legacy_rollout(self):
        config = GameConfig(grid_size=4, num_colors=3, rule_count=1,
                            frames=4, burn_in=1, stride=1)
        rule_seed = np.asarray(FIXTURE["rollout"]["rule_seed"], dtype=np.uint32)
        initial_seed = np.asarray(FIXTURE["rollout"]["initial_seed"], dtype=np.uint32)
        expected_initial = np.asarray(FIXTURE["rollout"]["initial_state"], dtype=np.int32)
        expected = np.asarray(FIXTURE["rollout"]["states"], dtype=np.int32)
        initial, states = reproduce(config, rule_seed, initial_seed)
        np.testing.assert_array_equal(initial, expected_initial)
        np.testing.assert_array_equal(states, expected)
        self.assertEqual(states.dtype, np.dtype("uint8"))
        self.assertEqual(expected.dtype, np.dtype("int32"))

    def test_gzip_matches_legacy_patch_stream(self):
        states = np.asarray(FIXTURE["tokenization"]["states"], dtype=np.uint8)
        self.assertAlmostEqual(gzip_ratio(states, 3),
                               FIXTURE["tokenization"]["gzip_ratio"])

    def test_packing_is_exact_legacy_tokenizer_output(self):
        states = np.asarray(FIXTURE["tokenization"]["states"], dtype=np.uint8)
        actual = pack_one(states, num_colors=3)
        torch.testing.assert_close(actual[0], torch.tensor(FIXTURE["tokenization"]["sequence"], dtype=torch.int32))
        torch.testing.assert_close(actual[1], torch.tensor(FIXTURE["tokenization"]["targets"], dtype=torch.int32))
        packed = PackedDataset(actual[0][None], actual[1][None], {"grid_len": 6})
        with tempfile.TemporaryDirectory() as temporary:
            restored = PackedDataset.load(packed.save(Path(temporary) / "packed.pt"))
            torch.testing.assert_close(restored.sequences, packed.sequences)
            torch.testing.assert_close(restored.targets, packed.targets)
            self.assertEqual(restored.metadata, packed.metadata)
        inputs, targets = packed.training_pairs(context_length=len(actual[0]) - 1)[0]
        torch.testing.assert_close(inputs, actual[0][:-1].long())
        reference_targets = actual[0].clone()
        reference_targets[:6] = -100
        torch.testing.assert_close(targets, reference_targets[1:].long())

    def test_checkpoint_roundtrip_reproduces_model_output_on_cpu(self):
        model_config = ModelConfig(vocab_size=83, context_length=11,
                                   layers=1, heads=2, embedding_dim=32)
        training_config = TrainingConfig(steps=1, batch_size=1, device="cpu")
        state = prepare(create_model(model_config, seed=9), model_config,
                        training_config, dataset_metadata={"source": "boundary-test"},
                        model_seed=9)
        sequence, token_targets = pack_one(
            np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4, 1) % 3,
            num_colors=3,
        )
        packed = PackedDataset(sequence[None], token_targets[None], {"grid_len": 6})
        pairs = packed.training_pairs(context_length=11)
        train(state, pairs, pairs)
        self.assertEqual(state.step, 1)
        self.assertIsNotNone(state.history[0]["validation_loss"])
        state.model.eval()
        tokens = torch.arange(11)[None] % model_config.vocab_size
        with torch.no_grad():
            expected = state.model(tokens)
        self.assertEqual(expected.shape, (1, 11, 83))
        self.assertEqual(trainable_parameters(state.model),
                         sum(p.numel() for p in state.model.parameters() if p.requires_grad))
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(state, Path(temporary))
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(set(raw), {"model", "optimizer", "scheduler", "epoch",
                                        "count", "best_val_loss", "best_val_loss_el",
                                        "metrics", "total_iterations"})
            restored = load_checkpoint(path)
            restored.model.eval()
            with torch.no_grad():
                actual = restored.model(tokens)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(asdict(restored.model_config), asdict(model_config))
            self.assertEqual(restored.dataset_metadata, {"source": "boundary-test"})
            self.assertEqual(restored.model_seed, 9)
            continued = load_checkpoint(path, training_config=TrainingConfig(
                steps=2, batch_size=1, device="cpu", checkpoint_interval=1))
            periodic = Path(temporary) / "periodic"
            train(continued, pairs, pairs, checkpoint_dir=periodic)
            self.assertEqual(continued.step, 2)
            self.assertEqual(len(continued.history), 2)
            self.assertTrue((periodic / "model_2_iter.pth").exists())
            downstream = create_model(ModelConfig(vocab_size=50, context_length=11,
                                                   layers=1, heads=2, embedding_dim=32))
            load_pretrained_body(downstream, path)
            torch.testing.assert_close(downstream.layers[0].self_attn.q_proj.weight,
                                       state.model.layers[0].self_attn.q_proj.weight)
            self.assertEqual(downstream.input_proj.weight.shape[0], 50)


if __name__ == "__main__":
    unittest.main()
