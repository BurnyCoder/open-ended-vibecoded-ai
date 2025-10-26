"""
Advanced Analysis: Deep dive into hypothesis dynamics

This example demonstrates:
1. Tracking hypothesis evolution through layers
2. Analyzing collapse entropy
3. Visualizing hypothesis specialization
4. Comparing different sequence patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import QuantumSuperpositionTransformer


def analyze_hypothesis_evolution(model, sequences, labels):
    """
    Track how hypotheses evolve across different input patterns.

    Args:
        model: Trained model
        sequences: List of input sequences to analyze
        labels: Labels for each sequence (for plotting)
    """
    print("\n" + "="*60)
    print("HYPOTHESIS EVOLUTION ANALYSIS")
    print("="*60)

    model.eval()
    all_results = []

    for seq, label in zip(sequences, labels):
        with torch.no_grad():
            logits, collapse_weights = model(seq.unsqueeze(0))

        # Calculate entropy at each layer
        entropies = []
        for weights in collapse_weights:
            w = weights[0].cpu().numpy()  # (seq_len, n_hyp)
            entropy = -np.sum(w * np.log(w + 1e-10), axis=1).mean()
            entropies.append(entropy)

        all_results.append({
            'label': label,
            'entropies': entropies,
            'collapse_weights': [w[0].cpu().numpy() for w in collapse_weights]
        })

        print(f"\n{label}:")
        print(f"  Layer-wise entropy: {[f'{e:.3f}' for e in entropies]}")

    return all_results


def visualize_hypothesis_evolution(results):
    """Create visualization of hypothesis evolution across layers."""

    n_results = len(results)
    n_layers = len(results[0]['entropies'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Entropy evolution
    ax1 = axes[0]
    for result in results:
        ax1.plot(range(1, n_layers + 1), result['entropies'],
                marker='o', label=result['label'], linewidth=2)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Collapse Entropy', fontsize=12)
    ax1.set_title('Hypothesis Uncertainty Across Layers', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, n_layers + 1))

    # Plot 2: Hypothesis dominance heatmap (last layer)
    ax2 = axes[1]

    # Average hypothesis weights across all sequences in final layer
    final_weights = np.array([r['collapse_weights'][-1].mean(axis=0)
                              for r in results])

    im = ax2.imshow(final_weights.T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Sequence Type', fontsize=12)
    ax2.set_ylabel('Hypothesis', fontsize=12)
    ax2.set_title('Final Layer Hypothesis Preferences', fontsize=14)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r['label'] for r in results], rotation=45, ha='right')
    ax2.set_yticks(range(final_weights.shape[1]))
    plt.colorbar(im, ax=ax2, label='Average Weight')

    plt.tight_layout()
    return fig


def compare_sequence_patterns():
    """Compare how model processes different sequence patterns."""

    print("\n" + "="*60)
    print("COMPARING DIFFERENT SEQUENCE PATTERNS")
    print("="*60)

    # Create model
    model = QuantumSuperpositionTransformer(
        vocab_size=50,
        d_model=64,
        n_hypotheses=4,
        n_layers=3,
        n_heads=4
    )

    # Generate different types of sequences
    seq_len = 12

    # Pattern 1: Repetitive
    repetitive = torch.tensor([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10])

    # Pattern 2: Ascending
    ascending = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])

    # Pattern 3: Random
    torch.manual_seed(42)
    random = torch.randint(0, 50, (seq_len,))

    # Pattern 4: Grouped
    grouped = torch.tensor([2, 2, 2, 15, 15, 15, 30, 30, 30, 45, 45, 45])

    sequences = [repetitive, ascending, random, grouped]
    labels = ['Repetitive', 'Ascending', 'Random', 'Grouped']

    # Analyze
    results = analyze_hypothesis_evolution(model, sequences, labels)

    # Visualize
    fig = visualize_hypothesis_evolution(results)

    output_path = 'hypothesis_evolution_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.close()

    return results


def analyze_collapse_confidence():
    """Analyze how confident the model is in its collapses."""

    print("\n" + "="*60)
    print("COLLAPSE CONFIDENCE ANALYSIS")
    print("="*60)

    model = QuantumSuperpositionTransformer(
        vocab_size=30,
        d_model=64,
        n_hypotheses=4,
        n_layers=3
    )

    # Create test sequences
    test_sequences = [
        torch.randint(0, 30, (16,)) for _ in range(10)
    ]

    all_entropies = []
    all_max_weights = []

    model.eval()
    with torch.no_grad():
        for seq in test_sequences:
            logits, collapse_weights = model(seq.unsqueeze(0))

            # Final layer collapse weights
            final_weights = collapse_weights[-1][0].cpu().numpy()

            # Calculate entropy (uncertainty)
            entropy = -np.sum(final_weights * np.log(final_weights + 1e-10), axis=1)

            # Maximum weight (confidence in dominant hypothesis)
            max_weight = final_weights.max(axis=1)

            all_entropies.extend(entropy)
            all_max_weights.extend(max_weight)

    print(f"\nCollapse Statistics (Final Layer):")
    print(f"  Mean entropy: {np.mean(all_entropies):.3f}")
    print(f"  Std entropy: {np.std(all_entropies):.3f}")
    print(f"  Mean max weight: {np.mean(all_max_weights):.3f}")
    print(f"  Std max weight: {np.std(all_max_weights):.3f}")

    # Create distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_entropies, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Entropy', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Collapse Entropy', fontsize=14)
    axes[0].axvline(np.mean(all_entropies), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(all_entropies):.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(all_max_weights, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Maximum Weight', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Dominant Hypothesis Weight', fontsize=14)
    axes[1].axvline(np.mean(all_max_weights), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(all_max_weights):.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = 'collapse_confidence_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.close()


def main():
    print("="*60)
    print("ADVANCED HYPOTHESIS DYNAMICS ANALYSIS")
    print("="*60)
    print("\nThis script performs deep analysis of how the")
    print("Quantum Superposition Transformer processes information")
    print("through multiple hypothesis spaces.")

    # Run analyses
    results = compare_sequence_patterns()
    analyze_collapse_confidence()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. Different sequence patterns may favor different hypotheses")
    print("2. Entropy decreases/increases show model uncertainty evolution")
    print("3. Maximum weights indicate confidence in dominant hypothesis")
    print("4. Layer-wise changes reveal information processing stages")
    print("\nGenerated visualizations:")
    print("  • hypothesis_evolution_analysis.png")
    print("  • collapse_confidence_analysis.png")


if __name__ == "__main__":
    main()
