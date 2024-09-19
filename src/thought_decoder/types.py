"""Type aliases for JAX arrays and batches."""

from typing import Annotated, TypeAlias

from jax import Array


Vector = Annotated[Array, 'Vector']
Batch: TypeAlias = tuple[Array, Array]
