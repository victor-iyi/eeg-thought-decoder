"""Agentic policy for the agent in the environment.

The agentic model helps learning for dynamic adaptation using Reinforcement Learning (RL) algorithms.

"""

from jax import Array
from flax import linen as nn


class PolicyNetwork(nn.Module):
    """Policy Network Module."""

    # The number of actions.
    action_dim: int

    def __call__(self, x: Array) -> Array:
        """Apply the policy network.

        Args:
            x (Array): The input tensor of shape (batch_size, input_dim).

        Returns:
            Array: The output tensor of shape (batch_size, action_dim).

        """
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.action_dim)(x)
        return logits  # actions probabilities.


class AgenticModel(nn.Module):
    """Agentic Model Module."""

    # The number of actions.
    action_dim: int

    def setup(self) -> None:
        """Set up the module."""
        self.policy_network = PolicyNetwork(action_dim=self.action_dim)

    def __call__(self, x: Array) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, input_dim).

        Returns:
            Array: The output tensor of shape (batch_size, action_dim).

        """
        logits = self.policy_network(x)
        action_probs = nn.softmax(logits)
        return action_probs
