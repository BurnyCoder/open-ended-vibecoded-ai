"""
Baseline Comparison: Quantum Superposition Transformer vs Standard Transformer

Research Questions:
1. How does our architecture compare to standard transformer?
2. What are the tradeoffs in performance vs complexity?
3. Does superposition provide benefits beyond parameter count?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import QuantumSuperpositionTransformer


class StandardTransformer(nn.Module):
    """
    Standard transformer for comparison.
    Matches parameter count as closely as possible.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Standard transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Embed and add position
        embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        # Transform
        transformed = self.transformer(embedded)

        # Project to vocabulary
        logits = self.output_proj(transformed)

        return logits


def train_and_evaluate(model, train_data, test_data, model_name, epochs=50):
    """Train model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = {
        'losses': [],
        'train_times': [],
        'test_accuracy': None,
        'final_loss': None,
        'total_params': sum(p.numel() for p in model.parameters())
    }

    print(f"Parameters: {metrics['total_params']:,}")

    # Training
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for seq in train_data:
            inputs = seq[:-1].unsqueeze(0)
            targets = seq[1:].unsqueeze(0)

            optimizer.zero_grad()

            if isinstance(model, QuantumSuperpositionTransformer):
                logits, _ = model(inputs)
            else:
                logits = model(inputs)

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                  targets.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_data)
        metrics['losses'].append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    total_time = time.time() - start_time
    metrics['train_times'] = total_time
    metrics['final_loss'] = metrics['losses'][-1]

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for seq in test_data:
            inputs = seq[:-1].unsqueeze(0)
            targets = seq[1:].unsqueeze(0)

            if isinstance(model, QuantumSuperpositionTransformer):
                logits, _ = model(inputs)
            else:
                logits = model(inputs)

            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()

    metrics['test_accuracy'] = correct / total

    print(f"\nFinal Results:")
    print(f"  Training time: {total_time:.2f}s")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Test accuracy: {metrics['test_accuracy']*100:.2f}%")

    return metrics


def generate_dataset(n_samples, seq_len, vocab_size, pattern='mixed'):
    """Generate training/test data."""
    sequences = []

    for i in range(n_samples):
        if pattern == 'mixed':
            pattern_type = ['repeat', 'arithmetic', 'random'][i % 3]
        else:
            pattern_type = pattern

        if pattern_type == 'repeat':
            pattern_len = np.random.randint(2, 5)
            p = torch.randint(0, vocab_size, (pattern_len,))
            seq = p.repeat((seq_len // pattern_len) + 1)[:seq_len]

        elif pattern_type == 'arithmetic':
            start = np.random.randint(0, vocab_size - seq_len)
            seq = torch.arange(start, start + seq_len) % vocab_size

        else:  # random
            seq = torch.randint(0, vocab_size, (seq_len,))

        sequences.append(seq)

    return torch.stack(sequences)


def compare_models():
    """Main comparison experiment."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON STUDY")
    print("="*70)

    vocab_size = 30
    seq_len = 16

    # Generate data
    print("\nGenerating datasets...")
    train_data = generate_dataset(100, seq_len, vocab_size, 'mixed')
    test_data = generate_dataset(30, seq_len, vocab_size, 'mixed')

    results = {}

    # Test 1: Similar parameter count
    print("\n" + "="*70)
    print("EXPERIMENT 1: Matched Parameter Count")
    print("="*70)

    qst = QuantumSuperpositionTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_hypotheses=4,
        n_layers=2,
        n_heads=4
    )

    std = StandardTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4
    )

    results['QST'] = train_and_evaluate(qst, train_data, test_data,
                                        "Quantum Superposition Transformer", epochs=50)
    results['Standard'] = train_and_evaluate(std, train_data, test_data,
                                             "Standard Transformer", epochs=50)

    # Test 2: Efficiency analysis
    print("\n" + "="*70)
    print("EXPERIMENT 2: Efficiency Analysis")
    print("="*70)

    configurations = [
        ('Small', 32, 1),
        ('Medium', 64, 2),
        ('Large', 96, 3)
    ]

    efficiency_results = []

    for config_name, d_model, n_layers in configurations:
        print(f"\n--- Configuration: {config_name} ---")

        qst_config = QuantumSuperpositionTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_hypotheses=4,
            n_layers=n_layers,
            n_heads=4
        )

        std_config = StandardTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4
        )

        qst_metrics = train_and_evaluate(qst_config, train_data[:50], test_data[:15],
                                        f"QST-{config_name}", epochs=30)
        std_metrics = train_and_evaluate(std_config, train_data[:50], test_data[:15],
                                        f"Std-{config_name}", epochs=30)

        efficiency_results.append({
            'config': config_name,
            'qst': qst_metrics,
            'std': std_metrics
        })

    return results, efficiency_results


def visualize_comparison(results, efficiency_results):
    """Create comparison visualizations."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results['QST']['losses'], label='QST', linewidth=2, color='#2E86AB')
    ax1.plot(results['Standard']['losses'], label='Standard', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    models = ['QST', 'Standard']
    accuracies = [results['QST']['test_accuracy'] * 100,
                  results['Standard']['test_accuracy'] * 100]
    colors = ['#2E86AB', '#A23B72']
    bars = ax2.bar(models, accuracies, color=colors, alpha=0.8)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Final Accuracy', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 100])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Parameter efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    params = [results['QST']['total_params'], results['Standard']['total_params']]
    bars = ax3.bar(models, params, color=colors, alpha=0.8)
    ax3.set_ylabel('Parameters', fontsize=11)
    ax3.set_title('Parameter Count', fontsize=13, fontweight='bold')
    ax3.ticklabel_format(style='plain', axis='y')

    # Plot 4: Training time
    ax4 = fig.add_subplot(gs[1, 1])
    times = [results['QST']['train_times'], results['Standard']['train_times']]
    bars = ax4.bar(models, times, color=colors, alpha=0.8)
    ax4.set_ylabel('Time (seconds)', fontsize=11)
    ax4.set_title('Training Time', fontsize=13, fontweight='bold')

    # Plot 5: Accuracy per million parameters
    ax5 = fig.add_subplot(gs[1, 2])
    efficiency = [results['QST']['test_accuracy'] * 100 / (results['QST']['total_params'] / 1e6),
                  results['Standard']['test_accuracy'] * 100 / (results['Standard']['total_params'] / 1e6)]
    bars = ax5.bar(models, efficiency, color=colors, alpha=0.8)
    ax5.set_ylabel('Accuracy % / Million Params', fontsize=11)
    ax5.set_title('Parameter Efficiency', fontsize=13, fontweight='bold')

    # Plot 6: Multi-configuration comparison
    ax6 = fig.add_subplot(gs[2, :])
    configs = [r['config'] for r in efficiency_results]
    qst_accs = [r['qst']['test_accuracy'] * 100 for r in efficiency_results]
    std_accs = [r['std']['test_accuracy'] * 100 for r in efficiency_results]

    x = np.arange(len(configs))
    width = 0.35

    ax6.bar(x - width/2, qst_accs, width, label='QST', color='#2E86AB', alpha=0.8)
    ax6.bar(x + width/2, std_accs, width, label='Standard', color='#A23B72', alpha=0.8)
    ax6.set_xlabel('Configuration', fontsize=11)
    ax6.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax6.set_title('Performance Across Configurations', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(configs)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    return fig


def print_summary(results, efficiency_results):
    """Print detailed comparison summary."""

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    # Main comparison
    print("\nMain Comparison (Matched Parameters):")
    print(f"\n  Quantum Superposition Transformer:")
    print(f"    Parameters: {results['QST']['total_params']:,}")
    print(f"    Final Loss: {results['QST']['final_loss']:.4f}")
    print(f"    Test Accuracy: {results['QST']['test_accuracy']*100:.2f}%")
    print(f"    Training Time: {results['QST']['train_times']:.2f}s")

    print(f"\n  Standard Transformer:")
    print(f"    Parameters: {results['Standard']['total_params']:,}")
    print(f"    Final Loss: {results['Standard']['final_loss']:.4f}")
    print(f"    Test Accuracy: {results['Standard']['test_accuracy']*100:.2f}%")
    print(f"    Training Time: {results['Standard']['train_times']:.2f}s")

    # Calculate differences
    acc_diff = (results['QST']['test_accuracy'] - results['Standard']['test_accuracy']) * 100
    time_ratio = results['QST']['train_times'] / results['Standard']['train_times']

    print(f"\n  Relative Performance:")
    if acc_diff > 0:
        print(f"    QST is {acc_diff:.2f}% more accurate")
    else:
        print(f"    Standard is {-acc_diff:.2f}% more accurate")

    if time_ratio > 1:
        print(f"    QST is {time_ratio:.2f}x slower to train")
    else:
        print(f"    QST is {1/time_ratio:.2f}x faster to train")

    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)

    print("\nKey Findings:")
    print("  1. Architectural tradeoffs:")
    if acc_diff > 2:
        print("     → QST shows significant accuracy advantage")
    elif acc_diff > 0:
        print("     → QST shows modest accuracy advantage")
    else:
        print("     → Standard transformer more accurate on this task")

    print("\n  2. Computational cost:")
    if time_ratio < 1.2:
        print("     → QST has comparable training time")
    elif time_ratio < 1.5:
        print("     → QST has moderately higher training cost")
    else:
        print("     → QST has significantly higher training cost")

    print("\n  3. When to use QST:")
    print("     → Tasks requiring explicit uncertainty modeling")
    print("     → Problems with multiple valid interpretations")
    print("     → Applications needing hypothesis tracking")


def main():
    print("\n" + "="*70)
    print("BASELINE COMPARISON RESEARCH")
    print("="*70)

    # Run comparisons
    results, efficiency_results = compare_models()

    # Visualize
    fig = visualize_comparison(results, efficiency_results)
    fig.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to: baseline_comparison.png")

    # Print summary
    print_summary(results, efficiency_results)

    plt.close('all')


if __name__ == "__main__":
    main()
