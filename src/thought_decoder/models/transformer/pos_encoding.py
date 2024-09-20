"""Positional encoding layer for Transformer model."""

import jax.numpy as jnp
from jax import Array
from flax import linen as nn


class PositionalEncoding(nn.Module):
    """Positional Encoding Layer."""

    # The model dimension.
    model_dim: int

    def __call__(self, x: Array) -> Array:
        """Add positional encoding to the input tensor.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            Array: The tensor with positional encoding added.

        """
        # pos_enc = jnp.arange(x.shape[1])[:, jnp.newaxis] / jnp.power(
        #     10000, 2 * jnp.arange(self.model_dim)[jnp.newaxis, :] / self.model_dim
        # )
        # pos_enc = jnp.where(jnp.arange(self.model_dim) % 2 == 0, jnp.sin(pos_enc), jnp.cos(pos_enc))
        # return x + pos_enc
        seq_len = x.shape[1]

        position = jnp.arange(seq_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.model_dim, 2) * -(jnp.log(10000.0) / self.model_dim))

        pe = jnp.zeros((seq_len, self.model_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[jnp.newaxis, ...]

        return x + pe
