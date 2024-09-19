# EEG Thought Decoder

[![CI](https://github.com/victor-iyi/eeg-thought-decoder/actions/workflows/ci.yaml/badge.svg)](https://github.com/victor-iyi/eeg-thought-decoder/actions/workflows/ci.yaml)
[![pre-commit](https://github.com/victor-iyi/eeg-thought-decoder/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/victor-iyi/eeg-thought-decoder/actions/workflows/pre-commit.yml)
[![formatter | docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![style | google](https://img.shields.io/badge/%20style-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

Decoding human thoughts from EEG signals is a complex task that requires capturing
intricate spatial and temporal patterns in the brain's electrical activity.

Recent advancements in AI, particularly in Transformer architectures and Large
Language Models (LLMs), have shown remarkable capabilities in modelling sequential
and complex data patterns.

In this exposition, I present a mathematically detailed proof that human thoughts
can be decoded from EEG signals using a sophisticated Transformer-based AI model.
I incorporate elements from Graph Neural Networks (GNNs),
expert models, and agentic models to enhance the model's specialization and accuracy.

## Mathematical Formulation

Let:

- $\mathbf{X} \in \mathbb{R}^{N \times T}$ denote the EEG data matrix,
where $N$ is the number of electrodes (channels) and $T$ is the number of time steps.
- $\mathbf{y} \in \mathbb{R}^C$ represent the target thought or cognitive state
encoded as a one-hot vector over $C$ possible classes.

Our goal is to find a function $f: \mathbb{R}^{N \times T} \rightarrow \mathbb{R}^C$
such that:

$\hat{\mathbf{y}} = f(\mathbf{X}),$

where $\hat{\mathbf{y}}$  is the model's prediction of the thought corresponding
to EEG input $\mathbf{X}$.

## Model Architecture

The proposed AI model integrates several advanced components:

1. **Transformer Encoder** for *Temporal Dynamics*
2. **Graph Neural Network** for *Spatial Relationships*
3. **Mixture of Experts** for *Specialization*
4. **Agentic Learning** for *Dynamic Adaptation*

### 1. Transformer Encoder for Temporal Dynamics

#### Input Embedding

Each EEG channel signal is embedded into a higher-dimensional space:

$\mathbf{E} = \text{Embedding}(\mathbf{X}) \in \mathbb{R}^{N \times T \times d_{\text{model}}},$

where $d_{\text{model}}$ is the model dimension.

#### Positional Encoding

To incorporate temporal information

$\mathbf{E}_{\text{pos}} = \mathbf{E} + \mathbf{P},$

where $\mathbf{P} \in \mathbb{R}^{N \times T \times d_{\text{model}}}$ is the
positional encoding matrix defined as:

$$\mathbf{P}{(n,t,2k)} = \sin\left( \frac{t}{10000^{2k/d{\text{model}}}} \right),

\quad
\mathbf{P}{(n,t,2k+1)} = \cos\left( \frac{t}{10000^{2k/d{\text{model}}}} \right).$$

#### Multi-Head Self-Attention

For each head $h$ and layer $l$:

- **Query**: $\mathbf{Q}h^{(l)} = \mathbf{E}{\text{pos}}^{(l)} \mathbf{W}_h^{Q(l)}$
- **Key**: $\mathbf{K}h^{(l)} = \mathbf{E}{\text{pos}}^{(l)} \mathbf{W}_h^{K(l)}$
- **Value**: $\mathbf{V}h^{(l)} = \mathbf{E}{\text{pos}}^{(l)} \mathbf{W}_h^{V(l)}$

Compute attention weights:

<!-- markdownlint-disable-next-line line-length -->
$\mathbf{A}_h^{(l)} = \text{softmax}\left( \frac{\mathbf{Q}_h^{(l)} (\mathbf{K}_h^{(l)})^\top}{\sqrt{d_k}} \right).$

Update embeddings:

$\mathbf{Z}_h^{(l)} = \mathbf{A}_h^{(l)} \mathbf{V}_h^{(l)}.$

Concatenate heads and apply linear transformation:

<!-- markdownlint-disable-next-line line-length -->
$\mathbf{Z}^{(l)} = \text{Concat}\left( \mathbf{Z}_1^{(l)}, \dots, \mathbf{Z}_H^{(l)} \right) \mathbf{W}^{O(l)}.$

- $\mathbf{Z}^{(l)}$: The output of the multi-head attention mechanism at layer
$l$.
- $Concat(…)$ : A function that concatenates the outputs from all $H$ attention heads.
- $\mathbf{Z}_1^{(l)}, \mathbf{Z}_2^{(l)}, …, \mathbf{Z}_H^{(l)}$: The outputs
from each of the $H$ attention heads at layer.
- $\mathbf{W}^{O(l)}$: The output weight matrix at layer $l$.

#### Feed-Forward Network

Apply position-wise feed-forward network:

<!-- markdownlint-disable-next-line line-length -->
$\mathbf{F}^{(l)} = \text{ReLU}\left( \mathbf{Z}^{(l)} \mathbf{W}_1^{(l)} + \mathbf{b}_1^{(l)} \right) \mathbf{W}_2^{(l)} + \mathbf{b}_2^{(l)}.$

#### Layer Normalization and Residual Connections

Each sub-layer includes residual connections and layer normalization:

<!-- markdownlint-disable-next-line line-length -->
$\mathbf{E}{\text{pos}}^{(l+1)} = \text{LayerNorm}\left( \mathbf{E}{\text{pos}}^{(l)} + \mathbf{F}^{(l)} \right).$

### 2. Graph Neural Network for Spatial Relationships

#### Graph Construction

Construct a graph $G = (V, E)$ where:

- $V$ represents the set of EEG electrodes.
- $E$ represents edges based on physical proximity or functional connectivity.

#### Graph Laplacian

Compute the normalized graph Laplacian:

$\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2},$

where $\mathbf{A}$ is the adjacency matrix and $\mathbf{D}$ is the degree matrix.

#### Graph Convolution

Apply GNN to capture spatial dependencies:

$\mathbf{H}^{(k+1)} = \sigma\left( \mathbf{L} \mathbf{H}^{(k)} \mathbf{W}^{(k)} \right),$

with $\mathbf{H}^{(0)} = \mathbf{E}_{\text{pos}}^{(L)}$ (output from Transformer
encoder), and $\sigma$ is an activation function (e.g., ReLU).

### 3. Mixture of Experts for Specialization

#### Expert Models

Define $M$ expert models ${f_m}_{m=1}^M$, each specializing in different aspects
(e.g., frequency bands, cognitive tasks).

#### Gating Mechanism

Learn gating functions $g_m(\mathbf{X})$ to weight each expert's contribution:

$\hat{\mathbf{y}} = \sum_{m=1}^M g_m(\mathbf{X}) f_m(\mathbf{X}),$

subject to $\sum_{m=1}^M g_m(\mathbf{X}) = 1$ and $g_m(\mathbf{X}) \geq 0$.

### 4. Agentic Learning for Dynamic Adaptation

Incorporate an agent that interacts with the environment and adapts based on feedback.

#### Policy Network

Define a policy $\pi_\theta(a | \mathbf{X})$ where $a$ is an action
(e.g., adjusting model parameters).

#### Reward Function

Define a reward $R(\hat{\mathbf{y}}, \mathbf{y})$ based on decoding accuracy.

#### Optimization Objective

Maximize expected reward:

<!-- markdownlint-disable-next-line line-length -->
$\max_\theta \mathbb{E}_{\mathbf{X}, \mathbf{y}} \left[ R\left( \hat{\mathbf{y}}, \mathbf{y} \right) \right].$

Update parameters using policy gradients:

$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}\left[ R \right].$

### Training Procedure

1. **Loss Function**
    Use cross-entropy loss:
    $L = -\frac{1}{M} \sum_{i=1}^M \mathbf{y}^{(i)^\top} \log \hat{\mathbf{y}}^{(i)}.$

2. **Regularization**
    Include regularization terms to prevent overfitting:
    <!-- markdownlint-disable-next-line line-length -->
    $L_{\text{total}} = L + \lambda \left( \| \theta \|^2 + \sum_{m=1}^M \| g_m \|^2 \right).$

3. **Optimization Algorithm**
    Use Adam optimizer with gradients computed via backpropagation.

---

#### Mathematical Proof of Decoding Capability

#### Universal Approximation Theorem for Transformers

Transformers are capable of approximating any sequence-to-sequence function,
given sufficient model capacity.

- **Existence of Function** $f$**:**
    There exists a function $f$ such that:
    <!-- markdownlint-disable-next-line line-length -->
    $\mathbf{y} = f(\mathbf{X}) = f_{\text{agent}} \left( f_{\text{experts}} \left( f_{\text{GNN}} \left( f_{\text{Transformer}}(\mathbf{X}) \right) \right) \right).$

- **Approximation by the Model:**
    Given the model's capacity and proper training, $f$ can be approximated
    arbitrarily well.

#### Proof Sketch

1. **Transformer Encoder:**
    Captures temporal dependencies, approximating temporal mappings in EEG data.

2. **Graph Neural Network:**
    Models spatial relationships, capturing the spatial structure of EEG electrodes.

3. **Mixture of Experts:**
    Enhances specialization, allowing the model to approximate complex functions
    by combining simpler ones.

4. **Agentic Model:**
    Adapts dynamically, refining the approximation based on feedback.

---

#### Verification and Evaluation

1. **Cross-Validation**
    Implement k-fold cross-validation to assess generalization.

2. **Performance Metrics**
    <!-- markdownlint-disable-next-line line-length -->
    - **Accuracy:** $\text{Accuracy} = \frac{1}{M} \sum_{i=1}^M \mathbf{1}\{\hat{\mathbf{y}}^{(i)} = \mathbf{y}^{(i)}\}$.
    - **Precision, Recall, F1-Score:** Calculated per class.

3. **Statistical Significance**
    Perform hypothesis testing (e.g., permutation tests) to confirm that decoding
    performance is significantly better than chance.

4. **Ablation Studies**
    Evaluate the impact of each component (Transformer, GNN, experts, agentic
    learning) by systematically removing them and observing performance changes.

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (MIT)

This project is opened under the [MIT][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/eeg-thought-decoder
[issues]: https://github.com/victor-iyi/eeg-thought-decoder/issues
[license]: ./LICENSE
