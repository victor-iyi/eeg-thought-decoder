"""Evaluation script.

Handles the evaluation loop for the `EEGThoughtDecoder` model.

"""

# pylint: disable=not-callable
# mypy: disable-error-code="assignment,no-untyped-call"

import jax.numpy as jnp
from flax.training import checkpoints, train_state

from thought_decoder.logging import logger
from thought_decoder.data import EEGDataLoader
from thought_decoder.models import EEGThoughtDecoder
from thought_decoder.models.utils import AgenticParams, GNNParams, MixtureOfExpertsParams, TransformerParams


def evaluate() -> None:
    """Evaluate the model."""
    # Data loader.
    data_loader = EEGDataLoader(data_dir='data/test/', batch_size=32, shuffle=False)

    # Model intialization.
    transformer_params = TransformerParams(
        num_layers=4,
        model_dim=128,
        num_heads=8,
        diff=512,
        input_vocab_size=1_000,
        maximum_position_encoding=1_000,
        dropout_rate=0.1,
    )
    gnn_params = GNNParams(
        num_layers=2,
        hidden_dim=128,
        adjacency_matrix=jnp.ones((64, 64)),  # Placeholder adjacency matrix.
    )
    moe_params = MixtureOfExpertsParams(num_experts=4, expert_output_dim=128)
    agentic_params = AgenticParams(action_dim=10)

    model = EEGThoughtDecoder(
        transformer_params=transformer_params,
        gnn_params=gnn_params,
        moe_params=moe_params,
        agentic_params=agentic_params,
    )

    # Restore checkpoint.
    params = checkpoints.restore_checkpoint('checkpoints/', target=None)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=None)

    total_correct, total_samples = 0, 0

    for batch in data_loader.get_batches():
        logits = state.apply_fn({'params': state.params}, batch[0], training=False)

        predictions = jnp.argmax(logits, axis=-1)
        total_correct += jnp.sum(predictions == batch[1])
        total_samples += batch[1].shape[0]

    accuracy = total_correct / total_samples
    logger.info(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    evaluate()
