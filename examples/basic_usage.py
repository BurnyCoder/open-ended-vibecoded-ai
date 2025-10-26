"""
Basic usage example for Quantum-Inspired Superposition Transformer
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import (
    QuantumSuperpositionTransformer,
    visualize_superposition_dynamics
)


def main():
    print("="*60)
    print("Basic Usage Example")
    print("="*60)

    # Create model
    model = QuantumSuperpositionTransformer(
        vocab_size=100,        # Vocabulary size
        d_model=128,           # Model dimension
        n_hypotheses=4,        # Number of parallel hypotheses
        n_layers=3,            # Number of transformer layers
        n_heads=4,             # Attention heads per layer
        max_seq_len=64         # Maximum sequence length
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample input
    batch_size = 2
    seq_len = 16
    sample_input = torch.randint(0, 100, (batch_size, seq_len))

    print(f"\nInput shape: {sample_input.shape}")
    print(f"Sample sequence: {sample_input[0].tolist()}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, collapse_weights = model(sample_input)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Number of collapse weight layers: {len(collapse_weights)}")

    # Analyze hypothesis usage
    print("\n" + "="*60)
    print("Hypothesis Collapse Analysis")
    print("="*60)

    for layer_idx, weights in enumerate(collapse_weights):
        # weights: (batch, seq_len, n_hypotheses)
        weights_np = weights[0].cpu().numpy()  # First batch item

        # Calculate which hypothesis dominates
        dominant = weights_np.argmax(axis=-1)
        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Dominant hypotheses: {dominant.tolist()}")
        print(f"  Average weights: {weights_np.mean(axis=0).round(3).tolist()}")

    # Get predictions
    predictions = logits.argmax(dim=-1)
    print(f"\nPredicted next tokens: {predictions[0].tolist()}")

    print("\n" + "="*60)
    print("Visualization Example")
    print("="*60)

    # Create visualization
    vocab = {i: f"T{i}" for i in range(100)}
    fig = visualize_superposition_dynamics(model, sample_input[:1], vocab)

    print("\nVisualization created!")
    print("(In a real scenario, this would be saved or displayed)")


if __name__ == "__main__":
    main()
