"""End-to-end replay contract for the standalone generator."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from nca import GameConfig
from nca.generate import NCAGenerator, generate, reproduce, sample_rules
from nca.io import save_batches
from nca.analyze import measure
from nca.dataset import TrajectoryDataset, iter_trajectories, read_manifest


class ReplayTest(unittest.TestCase):
    def test_saved_worlds_have_independent_rules_and_replay_exactly(self):
        config = GameConfig(
            grid_size=4, num_colors=3, rule_count=2, trajectories_per_rule=1,
            frames=4, burn_in=1, batch_size=1, patch_size=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "worlds"
            rules = sample_rules(config)
            direct = list(generate(config, rules))
            facade = list(NCAGenerator(config).generate(rules))
            for old, new in zip(direct, facade, strict=True):
                np.testing.assert_array_equal(old.states, new.states)
                np.testing.assert_array_equal(old.initial_seeds, new.initial_seeds)
            NCAGenerator(config).save(output)
            composed_output = save_batches(config, Path(temporary) / "composed", rules, direct)
            self.assertEqual(read_manifest(output), read_manifest(composed_output))
            for original, composed in zip(iter_trajectories(output),
                                          iter_trajectories(composed_output), strict=True):
                np.testing.assert_array_equal(original["states"], composed["states"])
            manifest = read_manifest(output)
            worlds = list(iter_trajectories(output))
            dataset = TrajectoryDataset.load(output)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.metadata, manifest)
            np.testing.assert_array_equal(dataset[0]["states"], worlds[0]["states"])
            self.assertEqual(len(dataset.filter(lambda record: record["rule_id"] == 1)), 1)
            copied = dataset.save(Path(temporary) / "worlds-copy")
            self.assertEqual((copied / "manifest.json").read_bytes(),
                             (output / "manifest.json").read_bytes())
            np.testing.assert_array_equal(TrajectoryDataset.load(copied)[0]["states"],
                                          worlds[0]["states"])
            self.assertEqual(
                set(manifest),
                {"schema_version", "config", "implementation", "shape", "frame_times", "rules", "shards"},
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["rules"], "rules.npy")
            self.assertEqual(
                set(manifest["shards"][0]),
                {"states", "rule_ids", "initial_seeds", "initial_states", "count"},
            )
            self.assertEqual(manifest["shards"][0]["states"], "trajectories-000000.npy")
            self.assertEqual(np.load(output / "rules.npy").dtype, np.dtype("uint32"))
            self.assertEqual(np.load(output / "trajectories-000000.npy").dtype, np.dtype("uint8"))
            self.assertEqual(manifest["shape"], [4, 4, 4, 1])
            self.assertEqual(len(worlds), 2)
            self.assertFalse(np.array_equal(worlds[0]["rule_seed"], worlds[1]["rule_seed"]))
            for world in worlds:
                initial, states = reproduce(config, world["rule_seed"], world["initial_seed"])
                np.testing.assert_array_equal(initial, world["initial_state"])
                np.testing.assert_array_equal(states, world["states"])
                self.assertGreater(measure(states, config.num_colors)["gzip_ratio_paper_style"], 0)


if __name__ == "__main__":
    unittest.main()
