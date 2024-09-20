"""Mixture of Expert model.

This module implements the Mixture of Experts model for specialization.

"""

# pylint: disable=attribute-defined-outside-init
import jax.numpy as jnp
from jax import Array
from flax import linen as nn


class Expert(nn.Module):
    """Expert Module."""

    # The output dimension.
    output_dim: int

    def __call__(self, x: Array) -> Array:
        """Apply the expert network.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            Array: The output tensor of shape (batch_size, seq_len, output_dim).

        """
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x


class MixtureOfExperts(nn.Module):
    """Mixture of Experts Module."""

    # The number of experts.
    num_experts: int

    # The output dimension of the expert.
    expert_output_dim: int

    def setup(self) -> None:
        """Set up the module."""
        self.experts = [Expert(output_dim=self.expert_output_dim) for _ in range(self.num_experts)]
        self.gating_network = nn.Dense(features=self.num_experts)

    def __call__(self, x: Array) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            Array: The output tensor of shape (batch_size, seq_len, model_dim).

        """
        # # Compute the gate values.
        # gate_values = nn.softmax(self.gating_network(x), axis=-1)
        #
        # # Compute the expert outputs.
        # expert_outputs = [expert(x) for expert in self.experts]
        #
        # # Compute the mixture of expert output.
        # mixture_of_experts_output = sum(
        #     gate_values[:, :, i, None] * expert_output for i, expert_output in enumerate(expert_outputs)
        # )
        #
        # return mixture_of_experts_output
        gating_logits = self.gating_network(x)
        gating_weights = nn.softmax(gating_logits, axis=-1)  # Shape: (batch_size, num_experts)

        # Shape: (batch_size, num_experts, seq_len, expert_output_dim)
        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=1)

        gated_output = jnp.einsum('be,bed->bd', gating_weights, expert_outputs)

        return gated_output
