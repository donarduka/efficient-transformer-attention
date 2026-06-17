# Efficient Transformer Attention

A comparative study of three efficient attention mechanisms - **Longformer**, **Linformer**, and **Reformer** - against standard full self-attention, evaluated for the trade-off between computational efficiency and downstream task performance.

## Motivation

Standard Transformer self-attention scales quadratically with sequence length (O(n²) time and memory), making it impractical for long-sequence tasks. This project implements and benchmarks three approaches that reduce this cost via different mechanisms:

- **Longformer** - sliding-window local attention + selective global tokens (O(n))
- **Linformer** - low-rank projection of key/value matrices (O(n))
- **Reformer** - locality-sensitive hashing (LSH) to approximate attention (O(n log n))

Each variant is implemented as a drop-in replacement for standard self-attention within a shared Transformer encoder, allowing direct, controlled comparison under identical training conditions.

## Evaluation

All four implementations (baseline + 3 variants) are benchmarked on tasks from the [Long Range Arena](https://github.com/google-research/long-range-arena) benchmark, measuring:

- Task accuracy
- Inference time per forward pass
- Peak memory usage

across multiple sequence lengths with results averaged over multiple runs with fixed random seeds for reproducibility.

## Project structure

```
.
├── baseline/ # Standard full self-attention implementation
├── longformer/ # Sliding-window + global token attention
├── linformer/ # Low-rank projected attention
├── reformer/ # LSH-based attention
├── configs/ # Experiment configuration files (hyperparameters, task settings)
├── results/ # Logged metrics and generated plots
└── scripts/ # Training, evaluation, and visualisation scripts
```
## Stack

Python, PyTorch, Hugging Face Transformers, NumPy, Matplotlib.

## Status

In progress.

## License

MIT

