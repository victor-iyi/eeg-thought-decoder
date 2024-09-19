"""Training script.

Handles the training loop for the `EEGThoughtDecoder` model.

"""

import time
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state

from thought_decoder.logging import logger
from thought_decoder.types import KeyArray, Array, InputShape
from thought_decoder.data import EEGDataLoader
from thought_decoder.models import EEGThoughtDecoder
from thought_decoder.models.utils import AgenticParams, GNNParams, MixtureOfExpertsParams, TransformerParams


def create_train_state(
    rng: KeyArray, model: nn.Module, learning_rate: float, input_shape: InputShape
) -> train_state.TrainState:
    """Create the initial training state."""
    params = model.init(rng, jnp.ones(input_shape), training=True)
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


@jax.jit
def train_step(state: train_state.TrainState, batch: tuple[Array, Array]) -> tuple[train_state.TrainState, float]:
    """Train for a single step.

    Args:
        state (train_state.TrainState): The training state.
        batch (tuple[Array, Array]): The batch of inputs and targets.

    Returns:
        tuple[train_state.TrainState, float]: The updated training state and the loss.

    """

    def loss_fn(params: Mapping[str, Any]) -> tuple[float, float]:
        logits = state.apply_fn({'params': params}, batch[0], training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


def train() -> None:
    """Training loop."""
    # Hyperparameters.
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    input_shape = (-1, 100, 64)  # placeholder input shape.

    # Initialize the data loader.
    data_loader = EEGDataLoader(data_dir='data/', batch_size=batch_size)

    # Model intialization.
    rng = jax.random.PRNGKey(0)
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

    state = create_train_state(rng, model, learning_rate, input_shape)

    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in data_loader.get_batches():
            state, loss = train_step(state, batch)
        epoch_time = time.time() - start_time
        logger.info(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Time: {epoch_time:.2f} sec.')

        # Save checkpoints.
        checkpoints.save_checkpoint(ckpt_dir='checkpoints/', target=state.params, step=epoch)


if __name__ == '__main__':
    train()
