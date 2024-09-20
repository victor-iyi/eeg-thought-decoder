"""Graph Neural Network (GNN) model for capturing spatial relationships in EEG data."""

# pylint: disable=attribute-defined-outside-init
import jax.numpy as jnp
from jax import Array
from flax import linen as nn


class GraphConvolutionLayer(nn.Module):
    """Graph Convolution Layer."""

    # The output dimension.
    output_dim: int

    # The adjacency matrix.
    adjacency_matrix: Array

    def __call__(self, x: Array) -> Array:
        """Apply Graph convolution.

        Args:
            x (Array): The input tensor of shape (batch_size, num_nodes, input_dim).

        Returns:
            Array: The output tensor of shape (batch_size, num_nodes, output_dim).

        """
        # Compute the normalized Laplacian matrix.
        laplacian = self._compute_normalized_laplacian(self.adjacency_matrix)

        # Compute the graph convolution.
        return jnp.matmul(laplacian, x) @ self.param(
            'weights', nn.initializers.xavier_uniform(), (x.shape[-1], self.output_dim)
        )

    def _compute_normalized_laplacian(self, adjacency_matrix: Array) -> Array:
        """Compute the normalized Laplacian matrix.

        Args:
            adjacency_matrix (Array): The adjacency matrix of shape (num_nodes, num_nodes).

        Returns:
            Array: The normalized Laplacian matrix.

        """
        # Compute the degree matrix.
        # degree_matrix = jnp.sum(adjacency_matrix, axis=0)
        # degree_matrix = jnp.where(degree_matrix > 0, 1.0 / jnp.sqrt(degree_matrix), 0.0)
        #
        # # Compute the normalized Laplacian matrix.
        # normalized_laplacian = jnp.eye(adjacency_matrix.shape[0]) - degree_matrix * adjacency_matrix * degree_matrix
        #
        # return normalized_laplacian

        # Compute the degree matrix.
        d = jnp.sum(adjacency_matrix, axis=-1)

        # Compute the inverse square root of the degree matrix.
        d_inv_sqrt = jnp.power(d, -0.5)
        d_inv_sqrt = jnp.diag(d_inv_sqrt)

        # Compute the normalized Laplacian matrix.
        laplacian = jnp.eye(adjacency_matrix.shape[0]) - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt

        # Return the normalized Laplacian matrix.
        return laplacian


class GNN(nn.Module):
    """Graph Neural Network (GNN) model."""

    # The number of layers.
    num_layers: int

    # The hidden dimension.
    hidden_dim: int

    # The adjacency matrix.
    adjacency_matrix: Array

    def setup(self) -> None:
        """Set up the module."""
        # Initialize the Graph Convolution Layers.
        self.gcn_layers = [
            GraphConvolutionLayer(output_dim=self.hidden_dim, adjacency_matrix=self.adjacency_matrix)
            for _ in range(self.num_layers)
        ]
        self.relu = nn.relu

    def __call__(self, x: Array) -> Array:
        """Apply the GNN model.

        Args:
            x (Array): The input tensor of shape (batch_size, num_nodes, input_dim).

        Returns:
            Array: The output tensor of shape (batch_size, num_nodes, hidden_dim).

        """
        # Apply the Graph Convolution Layers.
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x)
            x = self.relu(x)

        # Return the output tensor.
        return x
