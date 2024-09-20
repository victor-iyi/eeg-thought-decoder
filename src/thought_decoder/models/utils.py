"""Utility functions for the project."""

from typing import NotRequired, TypedDict
from jax import Array


class TransformerParams(TypedDict):
    """Hyperparameters for the Transformer model."""

    num_layers: int
    model_dim: int
    num_heads: int
    diff: int
    input_vocab_size: int
    maximum_position_encoding: int
    dropout_rate: NotRequired[float]


class GNNParams(TypedDict):
    """Hyperparameters for the GNN model."""

    num_layers: int
    hidden_dim: int
    adjacency_matrix: Array


class MixtureOfExpertsParams(TypedDict):
    """Hyperparameters for the Mixture of Experts model."""

    num_experts: int
    expert_output_dim: int


class AgenticParams(TypedDict):
    """Hyperparameters for the Agentic model."""

    action_dim: int
