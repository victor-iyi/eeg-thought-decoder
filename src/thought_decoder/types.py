"""Type aliases for JAX arrays and batches."""

from typing import TypeAlias

from jax import Array


Batch: TypeAlias = tuple[Array, Array]
