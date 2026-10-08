"""Read-only screen of intrinsic rule coupling on saved 40–50 worlds.

Run: uv run --all-packages python progress/nca_search/screen_rule_coupling.py
The score is a local logit Jacobian at 32 fixed cells in the first saved frame.
It reconstructs rule weights from saved seeds; it never generates trajectories.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

import jax.numpy as jnp
import numpy as np

from nca.config import GameConfig
from nca.game import Game

from screen_attention_lags import selected_receipts
from screen_existing import SOURCE, entropy, spearman, within_gzip_spearman
from sample_patch_replication import horizontal_copy_excess
from screen_transformer_targets import patch_grid


def effective_kernel(kernel: np.ndarray) -> np.ndarray:
    """Undo jnp.pad(..., 1, mode='wrap') on the *channel* axis too."""
    assert kernel.shape == (3, 3, 12, 4)
    result = np.array(kernel[:, :, 1:11, :], copy=True)
    result[:, :, 9, :] += kernel[:, :, 0, :]
    result[:, :, 0, :] += kernel[:, :, 11, :]
    return result


def rule_coupling(game: Game, seed: np.ndarray, states: np.ndarray) -> tuple[float, float, float]:
    params = game.rule(jnp.asarray(seed, dtype=jnp.uint32))["params"]
    k = effective_kernel(np.asarray(params["Conv_0"]["kernel"]))
    b0 = np.asarray(params["Conv_0"]["bias"])
    w1 = np.asarray(params["Conv_1"]["kernel"])[0, 0]
    b1 = np.asarray(params["Conv_1"]["bias"])
    w2 = np.asarray(params["Conv_2"]["kernel"])[0, 0]

    # Deterministic spaced cell sample.  The gate is evaluated on the saved
    # frame, but the input-to-logit Jacobian is a property of the rule.
    s = states[0, :, :, 0].astype(np.intp)
    sites = np.linspace(0, 143, 32, dtype=np.intp)
    yy, xx = np.divmod(sites, 12)
    pre = np.broadcast_to(b0, (len(sites), 4)).copy()
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            colors = s[(yy + dy) % 12, (xx + dx) % 12]
            pre += k[dy + 1, dx + 1, colors]
    gates = ((pre @ w1 + b1) > 0).astype(np.float64)
    # [site, spatial offset, color, output color]; subtract color mean because
    # a one-hot color change has zero sum along the input-color axis.
    influence = np.einsum("xyic,scd,de->sxyie", k, np.broadcast_to(w1, (len(sites), *w1.shape)) * gates[:, None, :], w2)
    influence -= influence.mean(axis=3, keepdims=True)
    energy = np.square(influence).sum(axis=(0, 3, 4)) / len(sites)
    center = float(energy[1, 1])
    neighbor = float((energy.sum() - center) / 8)
    horizontal = float(energy[1, 0] + energy[1, 2])
    vertical = float(energy[0, 1] + energy[2, 1])
    return neighbor / (center + 1e-12), neighbor, np.log((horizontal + 1e-12) / (vertical + 1e-12))


def trajectory_features(states: np.ndarray, gzip: float) -> dict[str, float]:
    s = states[:, :, :, 0].astype(np.uint8)
    p = patch_grid(states)
    copy_direction = float((p[:, :, 1:] == p[:, :, :-1]).mean() -
                           (p[:, 1:, :] == p[:, :-1, :]).mean())
    pairs = np.bincount((s[:-1].astype(np.int16) * 10 + s[1:]).ravel(), minlength=100).reshape(10, 10)
    cond = sum(row.sum() / pairs.sum() * entropy(row) for row in pairs if row.sum())
    return {
        "gzip": float(gzip),
        "horizontal_copy_excess": horizontal_copy_excess(states),
        "horizontal_minus_vertical_copy": copy_direction,
        "lag2_recurrence": float((s[2:] == s[:-2]).mean()),
        "transition_mi": entropy(pairs.sum(axis=0)) - cond,
        "cell_change": float((s[1:] != s[:-1]).mean()),
        "neighbor_predictive_gain_bits": neighbor_predictive_gain(s),
    }


def neighbor_predictive_gain(s: np.ndarray) -> float:
    """Held-out gain of (center, one neighbor) over center-only cell prediction.

    Fit counts on transitions 0..17, score on 18..25.  Average four cardinal
    neighbors; fixed add-one smoothing, no fit to the held-out transitions.
    """
    center_train, target_train = s[:18].astype(np.intp), s[1:19].astype(np.intp)
    center_test, target_test = s[18:-1].astype(np.intp), s[19:].astype(np.intp)
    base = np.bincount((center_train * 10 + target_train).ravel(), minlength=100).reshape(10, 10)
    base_prob = (base + 1) / (base.sum(axis=1, keepdims=True) + 10)
    baseline = -np.log2(base_prob[center_test, target_test]).mean()
    gains = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_train = np.roll(center_train, (dy, dx), axis=(1, 2))
        neighbor_test = np.roll(center_test, (dy, dx), axis=(1, 2))
        context_train = center_train * 10 + neighbor_train
        context_test = center_test * 10 + neighbor_test
        counts = np.bincount((context_train * 10 + target_train).ravel(), minlength=1000).reshape(100, 10)
        probabilities = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 10)
        gains.append(baseline + np.log2(probabilities[context_test, target_test]).mean())
    return float(np.mean(gains))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    receipts = selected_receipts(args.first, args.stride)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, receipt in enumerate(receipts):
        grouped[receipt["dataset"]].append((index, receipt))
    out = [None] * len(receipts)
    game = None
    for dataset, members in sorted(grouped.items()):
        root = SOURCE / dataset
        config = GameConfig(**json.loads((root / "manifest.json").read_text())["config"])
        if game is None:
            game = Game(config)
        else:
            assert (config.num_colors, config.conv_channels, config.hidden_channels) == (10, 4, 16)
        worlds = np.load(root / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            states = worlds[row["row"]]
            ratio, neighbor_energy, direction = rule_coupling(game, np.asarray(row["rule_seed"]), states)
            out[index] = {"neighbor_center_ratio": ratio, "neighbor_energy": neighbor_energy,
                          "horizontal_vertical_log_ratio": direction,
                          **trajectory_features(states, row["gzip_ratio"])}
    assert all(row is not None for row in out)
    data = {key: np.asarray([row[key] for row in out]) for key in out[0]}
    print(f"worlds={len(out)}, reference first={args.first}, stride={args.stride}")
    for key in ("neighbor_center_ratio", "neighbor_energy", "horizontal_vertical_log_ratio"):
        print(key, "p10/median/p90", np.quantile(data[key], [.1, .5, .9]))
        for outcome in ("horizontal_copy_excess", "horizontal_minus_vertical_copy", "lag2_recurrence", "transition_mi", "gzip", "cell_change", "neighbor_predictive_gain_bits"):
            print(key, outcome, f"rho={spearman(data[key], data[outcome]):+.4f}",
                  f"within-gzip-rho={within_gzip_spearman(data[key], data[outcome], data['gzip']):+.4f}")


if __name__ == "__main__":
    main()
