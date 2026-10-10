"""Compatibility and replay checks for the opt-in recurrence generator."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nca import (GameConfig, HorizontalRecurrenceGenerator, NCAGenerator,
                 NeuralRuleGenerator, RuleDesign, RuleSearch, optimize_rule,
                 reproduce_saved)
from nca.dataset import TrajectoryDataset, read_manifest
from nca.generate import reproduce


class RecurrenceTest(unittest.TestCase):
    def test_zero_is_original_and_designed_replays_with_v1_layout(self):
        config = GameConfig(grid_size=4, num_colors=3, rule_count=2,
                            trajectories_per_rule=1, frames=4, burn_in=1,
                            batch_size=1, patch_size=2)
        original = NCAGenerator(config)
        zero = HorizontalRecurrenceGenerator(config, 0)
        rule = original.sample_rule()
        np.testing.assert_array_equal(original.generate_one(rule).states,
                                      zero.generate_one(rule).states)

        with tempfile.TemporaryDirectory() as temporary:
            normal_dir = original.save(Path(temporary) / "normal")
            designed_dir = HorizontalRecurrenceGenerator(config, 0.3).save(
                Path(temporary) / "designed")
            normal = read_manifest(normal_dir)
            designed = read_manifest(designed_dir)
            self.assertEqual(set(normal), set(designed))
            self.assertEqual(normal["config"], designed["config"])
            self.assertEqual(normal["shape"], designed["shape"])
            self.assertEqual(set(normal["implementation"]), set(designed["implementation"]))
            self.assertEqual(json.loads((designed_dir / "design.json").read_text())[
                "copy_probability"], 0.3)
            self.assertFalse((normal_dir / "design.json").exists())
            for directory in (normal_dir, designed_dir):
                data = TrajectoryDataset.load(directory)
                for index, world in enumerate(data):
                    initial, states = reproduce_saved(directory, index)
                    np.testing.assert_array_equal(initial, world["initial_state"])
                    np.testing.assert_array_equal(states, world["states"])
                    self.assertEqual(world["states"].dtype, np.uint8)
                    self.assertEqual(world["states"].shape, (4, 4, 4, 1))
            world = TrajectoryDataset.load(normal_dir)[0]
            np.testing.assert_array_equal(reproduce(config, world["rule_seed"],
                                                     world["initial_seed"])[1],
                                          reproduce_saved(normal_dir, 0)[1])
            copied = TrajectoryDataset.load(designed_dir).save(Path(temporary) / "copy")
            self.assertEqual((copied / "design.json").read_bytes(),
                             (designed_dir / "design.json").read_bytes())
            np.testing.assert_array_equal(reproduce_saved(copied, 0)[1],
                                          TrajectoryDataset.load(copied)[0]["states"])

    def test_invalid_probability_and_missing_sidecar_fail(self):
        config = GameConfig(grid_size=4, num_colors=3, rule_count=1, frames=2)
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                HorizontalRecurrenceGenerator(config, value)
        with tempfile.TemporaryDirectory() as temporary:
            directory = HorizontalRecurrenceGenerator(config, 0.2).save(
                Path(temporary) / "designed")
            (directory / "design.json").unlink()
            with self.assertRaises(ValueError):
                reproduce_saved(directory, 0)

    def test_optimized_neural_rule_saves_and_replays(self):
        config = GameConfig(grid_size=4, num_colors=4, rule_count=1,
                            frames=4, burn_in=1, batch_size=1)
        rule = optimize_rule(config, 17,
                             RuleDesign(horizontal_excess=.02, gzip_min=0,
                                        gzip_max=10, patch_entropy=3),
                             RuleSearch(generations=2, population=2,
                                        train_worlds=1, heldout_worlds=1))
        self.assertEqual(rule.weights.dtype, np.float32)
        self.assertEqual(rule.rule_seed.shape, (2,))
        self.assertEqual(len(rule.receipt["after_heldout"]["worlds"]), 1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = NeuralRuleGenerator(config, [rule]).save(Path(temporary) / "neural")
            self.assertEqual(read_manifest(directory)["schema_version"], 1)
            self.assertEqual(json.loads((directory / "design.json").read_text())[
                "schema"], "optimized-neural-rules-v1")
            self.assertEqual(np.load(directory / "optimized-rule-weights.npy").shape[0], 1)
            world = TrajectoryDataset.load(directory)[0]
            initial, states = reproduce_saved(directory, 0)
            np.testing.assert_array_equal(initial, world["initial_state"])
            np.testing.assert_array_equal(states, world["states"])
            (directory / "design.json").unlink()
            with self.assertRaises(ValueError):
                reproduce_saved(directory, 0)


if __name__ == "__main__":
    unittest.main()
