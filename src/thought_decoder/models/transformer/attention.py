"""Attention mechanism for the Transformer model."""

# pylint: disable=attribute-defined-outside-init
import jax.numpy as jnp
from jax import Array
from flax import linen as nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention Mechanism."""

    # The model dimension.
    model_dim: int

    # The number of heads.
    num_heads: int

    def setup(self) -> None:
        """Set up the module."""
        assert self.model_dim % self.num_heads == 0, 'model_dim must be divisible by num_heads'

        self.depth = self.model_dim // self.num_heads

        self.wq = nn.Dense(features=self.model_dim)  # Query
        self.wk = nn.Dense(features=self.model_dim)  # Key
        self.wv = nn.Dense(features=self.model_dim)  # Value
        self.dense = nn.Dense(features=self.model_dim)  # Final dense layer

    def __call__(self, x: Array) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).

        Returns:
            Array: The output tensor of shape (batch_size, seq_len, model_dim).

        """
        batch_size = x.shape[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scaled_attention = self._scaled_dot_product_attention(q, k, v)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.model_dim)

        return self.dense(scaled_attention)

    def _split_heads(self, x: Array, batch_size: int) -> Array:
        """Split the heads.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).
            batch_size (int): The batch size.

        Returns:
            Array: The reshaped tensor of shape (batch_size, num_heads, seq_len, depth).

        """
        return x.reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)

    def _scaled_dot_product_attention(self, q: Array, k: Array, v: Array) -> Array:
        """Scaled Dot Product Attention.

        Args:
            q (Array): The query tensor of shape (batch_size, num_heads, seq_len_q, depth).
            k (Array): The key tensor of shape (batch_size, num_heads, seq_len_k, depth).
            v (Array): The value tensor of shape (batch_size, num_heads, seq_len_v, depth).

        Returns:
            Array: The output tensor of shape (batch_size, num_heads, seq_len_q, depth).

        """
        # Compute the dot product.
        # matmul_qk = jnp.matmul(q, k, transpose_b=True)
        matmul_qk = jnp.matmul(q, k.transpose(0, 1, 3, 2))

        # Scale the dot product.
        dk = jnp.array(k.shape[-1], dtype=jnp.float32)
        scaled_attention_logits = matmul_qk / jnp.sqrt(dk)

        # Softmax on the last axis.
        attention_weights = nn.softmax(scaled_attention_logits, axis=-1)

        # Compute the output.
        output = jnp.matmul(attention_weights, v)

        return output
