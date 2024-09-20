"""Transofrmers encoder layer & model."""

# pylint: disable=attribute-defined-outside-init
from jax import Array
from flax import linen as nn

from thought_decoder.models.transformer.attention import MultiHeadSelfAttention
from thought_decoder.models.transformer.pos_encoding import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    # The model dimension.
    model_dim: int

    # The number of heads in the multi-head attention.
    num_heads: int

    # The feed forward network dimension.
    diff: int

    # The dropout rate.
    dropout_rate: float = 0.1

    def setup(self) -> None:
        """Set up the module."""
        # Multi-Head Self Attention and Feed Forward Network.
        self.mha = MultiHeadSelfAttention(model_dim=self.model_dim, num_heads=self.num_heads)
        self.ffn = nn.Sequential([nn.Dense(features=self.diff), nn.relu, nn.Dense(features=self.model_dim)])

        # Layer Normalization and Dropout.
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: Array, training: bool) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, model_dim).
            training (bool): Whether the model is training or not.

        Returns:
            Array: The output tensor of shape (batch_size, seq_len, model_dim).

        """
        # Multi-Head Self Attention
        attn_output = self.mha(x)
        attn_output = self.dropout(attn_output, deterministic=not training)
        out1 = self.layernorm1(x + attn_output)

        # Feed Forward Network.
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, deterministic=not training)
        out2: Array = self.layernorm2(out1 + ffn_output)

        # Return the output.
        return out2


class TransformerEncoder(nn.Module):
    """Transformer Encoder with multiple Transformer Encoder Layers."""

    # The number of layers in the encoder.
    num_layers: int

    # The model dimension.
    model_dim: int

    # The number of heads in the multi-head attention.
    num_heads: int

    # The feed forward network dimension.
    diff: int

    # The input vocabulary size.
    input_vocab_size: int

    # The maximum position encoding.
    maximum_position_encoding: int

    # The dropout rate.
    dropout_rate: float = 0.1

    def setup(self) -> None:
        """Set up the module."""
        # Initialize the embedding & positional encoding layer.
        self.embedding = nn.Embed(num_embeddings=self.input_vocab_size, features=self.model_dim)
        self.pos_encoding = PositionalEncoding(model_dim=self.model_dim)

        # Intialize the Transformer Encoder Layers.
        self.encoder_layers = [
            TransformerEncoderLayer(
                model_dim=self.model_dim, num_heads=self.num_heads, diff=self.diff, dropout_rate=self.droptout_rate
            )
            for _ in range(self.num_layers)
        ]

        # Dropout layer.
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: Array, training: bool) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len).
            training (bool): Whether the model is training or not.

        Returns:
            Array: The output tensor of shape (batch_size, seq_len, model_dim).

        """
        # Get the batch size and sequence length.
        # batch_size, seq_len = x.shape

        # Add positional encoding to the input tensor.
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Apply the dropout layer.
        x = self.dropout(x, deterministic=not training)

        # Apply the Transformer Encoder Layers.
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)

        # Return the output tensor [batch_size, seq_len, model_dim]
        return x
