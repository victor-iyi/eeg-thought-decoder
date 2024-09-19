"""Data Loader module.

This module handles the loading and pre-processing of EEG data for training and
evaluation.

"""

from collections.abc import Iterator

from pathlib import Path

import jax
import jax.numpy as jnp


class EEGDataLoader:
    """EEG Data Loader.

    Loads and pre-processes EEG data for training and evaluation.

    """

    def __init__(self, data_dir: Path | str, batch_size: int = 32, shuffle: bool = True) -> None:
        """Initialize the EEGDataLoader.

        Args:
            data_dir (Path | str): The path to the data directory.
            batch_size (int): The batch size for the data loader.
                Defaults to 32.
            shuffle (bool): Whether to shuffle the data.
                Defaults to True.

        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # TODO(victor-iyi): You might wanna get the data from a remote source.
        self._load_data()

    def get_batches(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Get an iterator over the data batches.

        Yields:
            tuple[Array, Array]: A tuple of input and target arrays.

        """
        dataset_size = self.inputs.shape[0]
        indices = jnp.arange(dataset_size)

        if self.shuffle:
            indices = jax.random.permutation(jax.random.PRNGKey(0), indices)

        for start_idx in range(0, dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            yield self.inputs[batch_indices], self.targets[batch_indices]

    def _load_data(self) -> None:
        """Load the EEG data."""
        self.inputs = jnp.load(self.data_dir / 'inputs.npy')
        self.targets = jnp.load(self.data_dir / 'targets.npy')
