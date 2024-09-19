"""This module combines Transformer, GNN, Mixture of Experts and Agentic Model into one architecture."""

# from collections.abc import Mapping

from jax import Array
from flax import linen as nn

from thought_decoder.models import AgenticModel, GNN, MixtureOfExperts, TransformerEncoder
from thought_decoder.models.utils import AgenticParams, GNNParams, MixtureOfExpertsParams, TransformerParams


class EEGThoughtDecoder(nn.Module):
    """EEG Thought Decoder Module."""

    transformer_params: TransformerParams  # Mapping[str, int | float]
    gnn_params: GNNParams  # Mapping[str, int | Array]
    moe_params: MixtureOfExpertsParams  # Mapping[str, int]
    agentic_params: AgenticParams  # Mapping[str, int]

    def setup(self) -> None:
        """Set up the module."""
        self.transformer = TransformerEncoder(**self.transformer_params)
        self.gnn = GNN(**self.gnn_params)
        self.moe = MixtureOfExperts(**self.moe_params)
        self.agent = AgenticModel(**self.agentic_params)
        self.classifier = nn.Dense(features=self.agentic_params.action_dim)

    def __call__(self, x: Array, training: bool) -> Array:
        """Apply the module.

        Args:
            x (Array): The input tensor of shape (batch_size, seq_len, input_dim).
            training (bool): Whether the model is training or not.

        Returns:
            Array: The output tensor of shape (batch_size, action_dim).

        """
        # Transformer Encoder for temporal dynamics.
        x = self.transformer(x, training)

        # GNN for spatial relationships.
        x = self.gnn(x)

        # Mixture of Experts for specialization.
        x = self.moe(x)

        # Agentic Model for dynamic adaptation.
        action_probs = self.agent(x)

        # Classifier.
        logits = self.classifier(action_probs)
        return logits
