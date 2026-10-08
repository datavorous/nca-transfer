"""Discrete stochastic NCA rule, adapted from the upstream implementation."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class NCANetwork(nn.Module):
    outputs: int
    conv_channels: int = 4
    hidden_channels: int = 16

    @nn.compact
    def __call__(self, cells):
        cells = jnp.pad(cells, 1, mode="wrap")
        cells = nn.Conv(features=self.conv_channels, kernel_size=(3, 3), padding="VALID")(cells)
        cells = nn.Conv(features=self.hidden_channels, kernel_size=(1, 1))(cells)
        cells = nn.relu(cells)
        return nn.Conv(features=self.outputs, kernel_size=(1, 1))(cells)


class Game:
    def __init__(self, config):
        self.config = config
        self.network = NCANetwork(
            config.num_colors * config.groups,
            config.conv_channels,
            config.hidden_channels,
        )

    def rule(self, seed):
        grid = self.config.grid_size
        channels = self.config.num_colors * self.config.groups
        _, network_seed = jax.random.split(seed)
        return self.network.init(network_seed, jnp.zeros((grid, grid, channels)))

    def initial_state(self, seed):
        c = self.config
        logits = jax.random.normal(seed, (c.groups, c.num_colors))
        logits = jnp.broadcast_to(logits, (c.grid_size, c.grid_size, c.groups, c.num_colors))
        return jax.random.categorical(seed, logits, axis=-1)

    def step(self, seed, state, rule):
        c = self.config
        one_hot = jax.nn.one_hot(state, c.num_colors)
        grid = c.grid_size
        cells = one_hot.reshape(grid, grid, c.groups * c.num_colors)
        logits = self.network.apply(rule, cells).reshape(grid, grid, c.groups, c.num_colors)
        return jax.random.categorical(
            seed, (logits + one_hot * c.identity_bias) / c.temperature, axis=-1
        )
