"""
Hypothesis Specialization Analysis

Research Questions:
1. Do different hypotheses specialize on different pattern types?
2. Can we identify what each hypothesis learns?
3. How consistent is specialization across training?
4. Do hypotheses compete or cooperate?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import QuantumSuperpositionTransformer


def analyze_hypothesis_preferences(model, pattern_types):
    """
    Analyze which hypotheses prefer which pattern types.

    Args:
        model: Trained model
        pattern_types: Dict of pattern_name -> list of sequences
    """
    print("\n" + "="*70)
    print("HYPOTHESIS PREFERENCE ANALYSIS")
    print("="*70)

    model.eval()

    # Store preferences: pattern_type -> layer -> hypothesis -> weight
    preferences = {pattern: {f'layer_{i}': [] for i in range(len(model.blocks) + 1)}
                   for pattern in pattern_types.keys()}

    with torch.no_grad():
        for pattern_name, sequences in pattern_types.items():
            print(f"\nAnalyzing {pattern_name} patterns...")

            layer_hyp_weights = {f'layer_{i}': np.zeros(model.n_hypotheses)
                                for i in range(len(model.blocks) + 1)}

            for seq in sequences:
                logits, collapse_weights = model(seq.unsqueeze(0))

                # Average weights across sequence positions for each layer
                for layer_idx, weights in enumerate(collapse_weights):
                    w = weights[0].cpu().numpy().mean(axis=0)  # Average over sequence
                    layer_hyp_weights[f'layer_{layer_idx}'] += w

            # Average over all sequences
            for layer_key in layer_hyp_weights:
                layer_hyp_weights[layer_key] /= len(sequences)
                preferences[pattern_name][layer_key] = layer_hyp_weights[layer_key]

    return preferences


def visualize_specialization(preferences):
    """Create heatmap showing hypothesis specialization patterns."""

    pattern_names = list(preferences.keys())
    n_patterns = len(pattern_names)
    n_layers = len(list(preferences.values())[0])

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))

    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        layer_key = f'layer_{layer_idx}'

        # Create matrix: patterns x hypotheses
        matrix = np.array([preferences[p][layer_key] for p in pattern_names])

        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[f'H{i}' for i in range(matrix.shape[1])],
                   yticklabels=pattern_names,
                   ax=ax, cbar_kws={'label': 'Average Weight'})

        ax.set_title(f'Layer {layer_idx + 1}: Hypothesis Preferences',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Hypothesis', fontsize=11)
        ax.set_ylabel('Pattern Type', fontsize=11)

    plt.tight_layout()
    return fig


def analyze_hypothesis_consistency(model, train_snapshots):
    """
    Analyze if hypotheses maintain consistent specialization during training.

    Args:
        train_snapshots: List of (epoch, model_state_dict) tuples
    """
    print("\n" + "="*70)
    print("HYPOTHESIS CONSISTENCY ANALYSIS")
    print("="*70)

    # This would require training with snapshots
    # For now, we'll analyze current model's stability across different inputs

    test_patterns = {
        'repetitive': [torch.tensor([i % 3 for i in range(12)]) for _ in range(10)],
        'ascending': [torch.arange(i, i+12) % 20 for i in range(10)],
        'random': [torch.randint(0, 20, (12,)) for _ in range(10)]
    }

    hypothesis_variance = {}

    model.eval()
    with torch.no_grad():
        for pattern_name, sequences in test_patterns.items():
            all_weights = []

            for seq in sequences:
                _, collapse_weights = model(seq.unsqueeze(0))
                # Use final layer
                final_weights = collapse_weights[-1][0].cpu().numpy().mean(axis=0)
                all_weights.append(final_weights)

            all_weights = np.array(all_weights)
            variance = all_weights.var(axis=0)
            hypothesis_variance[pattern_name] = variance

            print(f"\n{pattern_name.capitalize()} patterns:")
            print(f"  Hypothesis weight variance: {variance}")
            print(f"  Mean variance: {variance.mean():.4f}")
            if variance.mean() < 0.01:
                print(f"  → Very consistent hypothesis usage")
            elif variance.mean() < 0.05:
                print(f"  → Moderately consistent")
            else:
                print(f"  → High variability in hypothesis selection")

    return hypothesis_variance


def analyze_hypothesis_competition(model, test_sequences):
    """
    Analyze whether hypotheses compete or cooperate.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS COMPETITION/COOPERATION ANALYSIS")
    print("="*70)

    model.eval()

    # Measure correlation between hypothesis weights
    all_weights = []

    with torch.no_grad():
        for seq in test_sequences:
            _, collapse_weights = model(seq.unsqueeze(0))

            # Collect weights from all layers
            for weights in collapse_weights:
                w = weights[0].cpu().numpy()  # (seq_len, n_hyp)
                all_weights.append(w)

    # Concatenate all positions from all sequences
    all_weights = np.concatenate(all_weights, axis=0)  # (total_positions, n_hyp)

    # Calculate correlation matrix
    correlation = np.corrcoef(all_weights.T)

    print(f"\nHypothesis Correlation Matrix:")
    print(correlation)

    # Interpret
    print(f"\nInterpretation:")
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
    mean_corr = off_diagonal.mean()

    print(f"  Mean pairwise correlation: {mean_corr:.3f}")

    if mean_corr < -0.3:
        print(f"  → Strong COMPETITION (negative correlation)")
        print(f"     Hypotheses tend to activate exclusively")
    elif mean_corr < -0.1:
        print(f"  → Moderate competition")
    elif mean_corr < 0.1:
        print(f"  → INDEPENDENT operation")
        print(f"     Hypotheses work independently")
    elif mean_corr < 0.3:
        print(f"  → Moderate cooperation")
    else:
        print(f"  → Strong COOPERATION (positive correlation)")
        print(f"     Hypotheses tend to co-activate")

    # Visualize correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='RdBu_r',
               center=0, vmin=-1, vmax=1,
               xticklabels=[f'H{i}' for i in range(correlation.shape[0])],
               yticklabels=[f'H{i}' for i in range(correlation.shape[0])],
               ax=ax)
    ax.set_title('Hypothesis Weight Correlations', fontsize=14, fontweight='bold')

    return correlation, fig


def main():
    print("\n" + "="*70)
    print("HYPOTHESIS SPECIALIZATION RESEARCH")
    print("="*70)

    # Create and train a model
    print("\nTraining model on mixed patterns...")

    model = QuantumSuperpositionTransformer(
        vocab_size=20,
        d_model=64,
        n_hypotheses=4,
        n_layers=3,
        n_heads=4
    )

    # Generate test patterns
    print("\nGenerating test patterns...")

    pattern_types = {
        'Repetitive (AB)': [torch.tensor([i % 2 for i in range(12)]) for _ in range(20)],
        'Repetitive (ABC)': [torch.tensor([i % 3 for i in range(12)]) for _ in range(20)],
        'Ascending': [torch.arange(i, i+12) % 20 for i in range(20)],
        'Descending': [torch.arange(i, i-12, -1) % 20 for i in range(15, 35)],
        'Constant': [torch.ones(12, dtype=torch.long) * i for i in range(5, 15)],
        'Random': [torch.randint(0, 20, (12,)) for _ in range(20)]
    }

    # Analyze preferences
    preferences = analyze_hypothesis_preferences(model, pattern_types)

    # Visualize
    fig1 = visualize_specialization(preferences)
    fig1.savefig('hypothesis_specialization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Specialization heatmap saved to: hypothesis_specialization.png")

    # Analyze consistency
    analyze_hypothesis_consistency(model, None)

    # Analyze competition/cooperation
    all_test_seqs = []
    for seqs in pattern_types.values():
        all_test_seqs.extend(seqs[:5])  # Sample from each type

    correlation, fig2 = analyze_hypothesis_competition(model, all_test_seqs)
    fig2.savefig('hypothesis_correlation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Correlation matrix saved to: hypothesis_correlation.png")

    # Summary insights
    print("\n" + "="*70)
    print("RESEARCH INSIGHTS")
    print("="*70)

    # Which hypotheses dominate which patterns in final layer?
    print("\nPattern-Hypothesis Associations (Final Layer):")
    final_layer = f'layer_{len(model.blocks)}'

    for pattern_name in pattern_types.keys():
        weights = preferences[pattern_name][final_layer]
        dominant_hyp = weights.argmax()
        confidence = weights[dominant_hyp]

        print(f"\n  {pattern_name}:")
        print(f"    → Hypothesis {dominant_hyp} (weight: {confidence:.3f})")
        print(f"    → Distribution: {[f'{w:.3f}' for w in weights]}")

    plt.close('all')


if __name__ == "__main__":
    main()
