"""Configuration surface for the repository's existing Llama model."""

from dataclasses import dataclass

from sampleeff_shared.models import create_llama_model


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    layers: int = 4
    heads: int = 4
    embedding_dim: int = 128
    dropout: float = 0.1
    output_vocab_size: int | None = None
    tie_embeddings: bool = False

    @classmethod
    def original_nca(cls) -> "ModelConfig":
        """Published NCA launcher's model and sequence settings."""
        return cls(vocab_size=64_000, context_length=1024, layers=24,
                   heads=32, embedding_dim=2048, dropout=0.1)

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.layers, self.heads,
               self.embedding_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if (self.embedding_dim // self.heads) % 2:
            raise ValueError("rotary attention requires an even head dimension")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def create_model(config: ModelConfig, seed: int | None = None):
    if seed is not None:
        import torch

        torch.manual_seed(seed)
    return create_llama_model(
        vocab_size=config.vocab_size,
        seq_length=config.context_length,
        n_layer=config.layers,
        n_head=config.heads,
        n_embd=config.embedding_dim,
        output_vocab=config.output_vocab_size,
        dropout=config.dropout,
        weight_tying=config.tie_embeddings,
    )


def trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
