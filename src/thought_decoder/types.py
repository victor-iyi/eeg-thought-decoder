"""Type aliases for JAX arrays and batches."""

from typing import Annotated, TypeAlias

from jax import Array


KeyArray: TypeAlias = Annotated[Array, 'PRNGKey']
Vector: TypeAlias = Annotated[Array, 'Vector']
Batch: TypeAlias = tuple[Array, Array]
InputShape: TypeAlias = Annotated[tuple[int, int, int], 'batch_size, seq_len, input_dim']
