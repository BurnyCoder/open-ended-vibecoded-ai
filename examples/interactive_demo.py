"""
Interactive Demo: Explore hypothesis collapse in real-time

This script lets you input sequences and see how hypotheses collapse.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import QuantumSuperpositionTransformer


def print_hypothesis_analysis(collapse_weights, n_hypotheses):
    """Pretty print hypothesis collapse analysis."""

    print("\n" + "-"*60)
    print("HYPOTHESIS COLLAPSE ANALYSIS")
    print("-"*60)

    for layer_idx, weights in enumerate(collapse_weights):
        w = weights[0].cpu().numpy()  # (seq_len, n_hyp)

        print(f"\nLayer {layer_idx + 1}:")

        # Average weights
        avg_weights = w.mean(axis=0)
        print(f"  Average weights: {[f'{x:.3f}' for x in avg_weights]}")

        # Dominant hypothesis
        dominant = w.argmax(axis=-1)
        unique, counts = np.unique(dominant, return_counts=True)
        print(f"  Dominant hypothesis distribution:")
        for hyp, count in zip(unique, counts):
            print(f"    Hypothesis {hyp}: {count}/{len(dominant)} positions ({count/len(dominant)*100:.1f}%)")

        # Entropy (uncertainty)
        entropy = -np.sum(w * np.log(w + 1e-10), axis=1).mean()
        print(f"  Average entropy: {entropy:.3f} ", end="")
        if entropy < 0.5:
            print("(very certain)")
        elif entropy < 1.0:
            print("(confident)")
        elif entropy < 1.3:
            print("(moderate uncertainty)")
        else:
            print("(high uncertainty)")


def run_interactive_demo():
    """Interactive demo for exploring the model."""

    print("="*60)
    print("QUANTUM SUPERPOSITION TRANSFORMER - INTERACTIVE DEMO")
    print("="*60)

    # Create model
    vocab_size = 100
    model = QuantumSuperpositionTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_hypotheses=4,
        n_layers=3,
        n_heads=4
    )

    model.eval()

    print(f"\nModel initialized:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: 128")
    print(f"  Number of hypotheses: 4")
    print(f"  Number of layers: 3")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*60)
    print("DEMO MODE")
    print("="*60)
    print("\nYou can:")
    print("  1. Enter a custom sequence (e.g., '1,5,3,8,2')")
    print("  2. Try preset patterns:")
    print("     - 'repetitive': Alternating pattern")
    print("     - 'ascending': Increasing sequence")
    print("     - 'random': Random numbers")
    print("     - 'grouped': Clustered values")
    print("  3. Type 'quit' to exit")

    while True:
        print("\n" + "-"*60)
        user_input = input("\nEnter sequence or pattern name (or 'quit'): ").strip()

        if user_input.lower() == 'quit':
            print("\nExiting demo. Thank you!")
            break

        # Parse input
        try:
            if user_input.lower() == 'repetitive':
                sequence = torch.tensor([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10])
                print(f"Using repetitive pattern: {sequence.tolist()}")

            elif user_input.lower() == 'ascending':
                sequence = torch.arange(0, 24, 2)
                print(f"Using ascending pattern: {sequence.tolist()}")

            elif user_input.lower() == 'random':
                torch.manual_seed(np.random.randint(0, 10000))
                sequence = torch.randint(0, vocab_size, (12,))
                print(f"Using random pattern: {sequence.tolist()}")

            elif user_input.lower() == 'grouped':
                sequence = torch.tensor([2, 2, 2, 15, 15, 15, 30, 30, 30, 45, 45, 45])
                print(f"Using grouped pattern: {sequence.tolist()}")

            else:
                # Parse custom sequence
                tokens = [int(x.strip()) for x in user_input.split(',')]
                if any(t < 0 or t >= vocab_size for t in tokens):
                    print(f"Error: All tokens must be between 0 and {vocab_size-1}")
                    continue
                sequence = torch.tensor(tokens)
                print(f"Using custom sequence: {sequence.tolist()}")

        except ValueError:
            print("Error: Invalid input. Use comma-separated integers or a pattern name.")
            continue

        # Process sequence
        with torch.no_grad():
            logits, collapse_weights = model(sequence.unsqueeze(0))
            predictions = logits.argmax(dim=-1)[0]

        print(f"\nSequence length: {len(sequence)}")
        print(f"Input sequence:  {sequence.tolist()}")
        print(f"Predicted next:  {predictions.tolist()}")

        # Show hypothesis analysis
        print_hypothesis_analysis(collapse_weights, model.n_hypotheses)


def quick_comparison():
    """Quick comparison of different patterns."""

    print("\n" + "="*60)
    print("QUICK PATTERN COMPARISON")
    print("="*60)

    model = QuantumSuperpositionTransformer(
        vocab_size=50,
        d_model=64,
        n_hypotheses=4,
        n_layers=3
    )

    patterns = {
        'Repetitive': torch.tensor([5, 10, 5, 10, 5, 10, 5, 10]),
        'Ascending': torch.arange(0, 16, 2),
        'Random': torch.randint(0, 50, (8,)),
        'Constant': torch.ones(8, dtype=torch.long) * 7
    }

    model.eval()

    for name, seq in patterns.items():
        with torch.no_grad():
            logits, collapse_weights = model(seq.unsqueeze(0))

        # Calculate final layer entropy
        final_weights = collapse_weights[-1][0].cpu().numpy()
        entropy = -np.sum(final_weights * np.log(final_weights + 1e-10), axis=1).mean()

        # Dominant hypothesis
        dominant = final_weights.argmax(axis=-1)
        unique, counts = np.unique(dominant, return_counts=True)
        most_common = unique[counts.argmax()]

        print(f"\n{name}:")
        print(f"  Sequence: {seq.tolist()}")
        print(f"  Final entropy: {entropy:.3f}")
        print(f"  Most dominant hypothesis: {most_common}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Interactive demo of Quantum Superposition Transformer')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick comparison instead of interactive mode')

    args = parser.parse_args()

    if args.quick:
        quick_comparison()
    else:
        run_interactive_demo()
