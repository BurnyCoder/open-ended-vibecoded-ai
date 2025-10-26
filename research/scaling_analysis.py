"""
Scaling Analysis: How does the architecture scale with different parameters?

Research Questions:
1. How does performance change with number of hypotheses?
2. What is the impact of model depth?
3. How does embedding dimension affect learning?
4. What is the computational cost vs performance tradeoff?
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_superposition_transformer import QuantumSuperpositionTransformer


def generate_pattern_dataset(n_samples=100, seq_len=16, vocab_size=20, pattern_type='repeat'):
    """Generate synthetic dataset with clear patterns."""
    sequences = []

    for _ in range(n_samples):
        if pattern_type == 'repeat':
            # Repeating pattern
            pattern_len = np.random.randint(2, 5)
            pattern = torch.randint(0, vocab_size, (pattern_len,))
            seq = pattern.repeat((seq_len // pattern_len) + 1)[:seq_len]

        elif pattern_type == 'arithmetic':
            # Arithmetic sequence
            start = np.random.randint(0, vocab_size - seq_len)
            step = np.random.choice([1, 2])
            seq = torch.arange(start, start + seq_len * step, step) % vocab_size

        elif pattern_type == 'fibonacci':
            # Fibonacci-like
            a, b = np.random.randint(0, 5), np.random.randint(0, 5)
            seq_list = [a, b]
            for _ in range(seq_len - 2):
                seq_list.append((seq_list[-1] + seq_list[-2]) % vocab_size)
            seq = torch.tensor(seq_list)

        else:  # random
            seq = torch.randint(0, vocab_size, (seq_len,))

        sequences.append(seq)

    return torch.stack(sequences)


def train_model(model, train_data, epochs=50, lr=0.001, verbose=False):
    """Train model and return loss curve."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for seq in train_data:
            inputs = seq[:-1].unsqueeze(0)
            targets = seq[1:].unsqueeze(0)

            optimizer.zero_grad()
            logits, _ = model(inputs)

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                  targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        losses.append(avg_loss)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return losses


def evaluate_model(model, test_data):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for seq in test_data:
            inputs = seq[:-1].unsqueeze(0)
            targets = seq[1:].unsqueeze(0)

            logits, _ = model(inputs)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == targets).sum().item()
            total += targets.numel()

    return correct / total


def experiment_num_hypotheses():
    """Experiment: Impact of number of hypotheses."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Number of Hypotheses")
    print("="*70)

    hypothesis_counts = [1, 2, 4, 8]
    vocab_size = 20
    seq_len = 16

    results = {
        'n_hypotheses': [],
        'final_loss': [],
        'accuracy': [],
        'training_time': [],
        'parameters': []
    }

    # Generate dataset
    train_data = generate_pattern_dataset(50, seq_len, vocab_size, 'repeat')
    test_data = generate_pattern_dataset(20, seq_len, vocab_size, 'repeat')

    for n_hyp in hypothesis_counts:
        print(f"\nTesting with {n_hyp} hypotheses...")

        model = QuantumSuperpositionTransformer(
            vocab_size=vocab_size,
            d_model=64,
            n_hypotheses=n_hyp,
            n_layers=2,
            n_heads=4
        )

        n_params = sum(p.numel() for p in model.parameters())

        # Train
        start_time = time.time()
        losses = train_model(model, train_data, epochs=30, lr=0.001)
        train_time = time.time() - start_time

        # Evaluate
        accuracy = evaluate_model(model, test_data)

        results['n_hypotheses'].append(n_hyp)
        results['final_loss'].append(losses[-1])
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['parameters'].append(n_params)

        print(f"  Parameters: {n_params:,}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Training Time: {train_time:.2f}s")

    return results


def experiment_model_depth():
    """Experiment: Impact of model depth."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Model Depth")
    print("="*70)

    layer_counts = [1, 2, 3, 4]
    vocab_size = 20
    seq_len = 16

    results = {
        'n_layers': [],
        'final_loss': [],
        'accuracy': [],
        'training_time': [],
        'parameters': []
    }

    train_data = generate_pattern_dataset(50, seq_len, vocab_size, 'arithmetic')
    test_data = generate_pattern_dataset(20, seq_len, vocab_size, 'arithmetic')

    for n_layers in layer_counts:
        print(f"\nTesting with {n_layers} layers...")

        model = QuantumSuperpositionTransformer(
            vocab_size=vocab_size,
            d_model=64,
            n_hypotheses=4,
            n_layers=n_layers,
            n_heads=4
        )

        n_params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        losses = train_model(model, train_data, epochs=30, lr=0.001)
        train_time = time.time() - start_time

        accuracy = evaluate_model(model, test_data)

        results['n_layers'].append(n_layers)
        results['final_loss'].append(losses[-1])
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['parameters'].append(n_params)

        print(f"  Parameters: {n_params:,}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Training Time: {train_time:.2f}s")

    return results


def experiment_embedding_dimension():
    """Experiment: Impact of embedding dimension."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Impact of Embedding Dimension")
    print("="*70)

    d_models = [32, 64, 128, 256]
    vocab_size = 20
    seq_len = 16

    results = {
        'd_model': [],
        'final_loss': [],
        'accuracy': [],
        'training_time': [],
        'parameters': []
    }

    train_data = generate_pattern_dataset(50, seq_len, vocab_size, 'fibonacci')
    test_data = generate_pattern_dataset(20, seq_len, vocab_size, 'fibonacci')

    for d_model in d_models:
        print(f"\nTesting with d_model={d_model}...")

        model = QuantumSuperpositionTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_hypotheses=4,
            n_layers=2,
            n_heads=4
        )

        n_params = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        losses = train_model(model, train_data, epochs=30, lr=0.001)
        train_time = time.time() - start_time

        accuracy = evaluate_model(model, test_data)

        results['d_model'].append(d_model)
        results['final_loss'].append(losses[-1])
        results['accuracy'].append(accuracy)
        results['training_time'].append(train_time)
        results['parameters'].append(n_params)

        print(f"  Parameters: {n_params:,}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Training Time: {train_time:.2f}s")

    return results


def visualize_scaling_results(results_hyp, results_depth, results_dim):
    """Create comprehensive visualization of scaling experiments."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Accuracy
    axes[0, 0].plot(results_hyp['n_hypotheses'],
                    [a*100 for a in results_hyp['accuracy']],
                    'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Number of Hypotheses', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title('Impact of Hypothesis Count', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(results_depth['n_layers'],
                    [a*100 for a in results_depth['accuracy']],
                    'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Number of Layers', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Impact of Model Depth', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(results_dim['d_model'],
                    [a*100 for a in results_dim['accuracy']],
                    'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[0, 2].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0, 2].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 2].set_title('Impact of Embedding Size', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Parameter efficiency (accuracy per million parameters)
    axes[1, 0].plot(results_hyp['n_hypotheses'],
                    [a*100 / (p/1e6) for a, p in zip(results_hyp['accuracy'],
                                                       results_hyp['parameters'])],
                    'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[1, 0].set_xlabel('Number of Hypotheses', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy % per Million Params', fontsize=11)
    axes[1, 0].set_title('Parameter Efficiency', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(results_depth['n_layers'],
                    [a*100 / (p/1e6) for a, p in zip(results_depth['accuracy'],
                                                       results_depth['parameters'])],
                    'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[1, 1].set_xlabel('Number of Layers', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy % per Million Params', fontsize=11)
    axes[1, 1].set_title('Parameter Efficiency', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(results_dim['d_model'],
                    [a*100 / (p/1e6) for a, p in zip(results_dim['accuracy'],
                                                       results_dim['parameters'])],
                    'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 2].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1, 2].set_ylabel('Accuracy % per Million Params', fontsize=11)
    axes[1, 2].set_title('Parameter Efficiency', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*70)
    print("SCALING ANALYSIS: Quantum Superposition Transformer")
    print("="*70)
    print("\nInvestigating how architectural choices affect performance\n")

    # Run experiments
    results_hyp = experiment_num_hypotheses()
    results_depth = experiment_model_depth()
    results_dim = experiment_embedding_dimension()

    # Visualize
    fig = visualize_scaling_results(results_hyp, results_depth, results_dim)

    output_path = 'scaling_analysis_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print("\n1. NUMBER OF HYPOTHESES:")
    best_hyp_idx = np.argmax(results_hyp['accuracy'])
    print(f"   Best: {results_hyp['n_hypotheses'][best_hyp_idx]} hypotheses")
    print(f"   Accuracy: {results_hyp['accuracy'][best_hyp_idx]*100:.2f}%")
    print(f"   Observation: ", end="")
    if results_hyp['accuracy'][-1] > results_hyp['accuracy'][0]:
        print("More hypotheses improve performance")
    else:
        print("Diminishing returns with more hypotheses")

    print("\n2. MODEL DEPTH:")
    best_depth_idx = np.argmax(results_depth['accuracy'])
    print(f"   Best: {results_depth['n_layers'][best_depth_idx]} layers")
    print(f"   Accuracy: {results_depth['accuracy'][best_depth_idx]*100:.2f}%")
    print(f"   Observation: ", end="")
    if results_depth['accuracy'][-1] > results_depth['accuracy'][0] * 1.1:
        print("Deeper models significantly better")
    else:
        print("Moderate depth sufficient for this task")

    print("\n3. EMBEDDING DIMENSION:")
    best_dim_idx = np.argmax(results_dim['accuracy'])
    print(f"   Best: d_model={results_dim['d_model'][best_dim_idx]}")
    print(f"   Accuracy: {results_dim['accuracy'][best_dim_idx]*100:.2f}%")
    print(f"   Observation: ", end="")
    if results_dim['accuracy'][-1] > results_dim['accuracy'][-2]:
        print("Larger dimensions beneficial")
    else:
        print("Sweet spot found at medium dimension")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
