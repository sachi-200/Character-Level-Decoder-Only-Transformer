# Character-Level Decoder-Only Transformer

A lightweight decoder-only Transformer implemented from scratch in **PyTorch** for character-level language modeling. The project demonstrates the core components of modern Transformer architectures, compares **Pre-LayerNorm** and **Post-LayerNorm** training strategies, generates text through autoregressive decoding, and visualizes self-attention patterns learned by the model.

---

## Features

- Decoder-only Transformer architecture
- Character-level tokenization
- Learned token and positional embeddings
- Multi-Head Self-Attention with causal masking
- Position-wise Feed-Forward Network using GELU activation
- Configurable Layer Normalization
  - Pre-LayerNorm
  - Post-LayerNorm
- Residual connections
- Temperature-controlled text generation
- Attention heatmap visualization
- Automatic model parameter count

---

## Model Architecture

| Hyperparameter | Value |
|---------------|------:|
| Embedding Dimension (`d_model`) | 64 |
| Attention Heads | 4 |
| Transformer Layers | 3 |
| Context Window | 128 characters |
| Feed-Forward Dimension | 256 |
| Dropout | 0.1 |
| Activation | GELU |
| Optimizer | Adam |
| Learning Rate | 3 × 10⁻⁴ |

The model uses **causal self-attention**, ensuring each token can attend only to previously generated tokens, making it suitable for autoregressive language modeling.

---

## Project Components

### Character Embeddings

Each input character is mapped to a learned embedding vector. Learned positional embeddings are added to preserve sequence order.

### Multi-Head Self-Attention

Implements masked multi-head attention using a causal mask so that predictions depend only on previous characters.

### Feed-Forward Network

Each Transformer block contains a position-wise feed-forward network consisting of:

- Linear layer
- GELU activation
- Linear layer

### Transformer Blocks

Each block includes:

- Multi-head self-attention
- Feed-forward network
- Residual connections
- Layer Normalization

Both **Pre-LayerNorm** and **Post-LayerNorm** variants are implemented for comparison.

---
