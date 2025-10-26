"""
Quantum-Inspired Superposition Transformer

A novel neural architecture that maintains multiple competing hypotheses in superposition,
using quantum-inspired mechanics for information processing.

Key innovations:
1. Superposition States: Multiple parallel computation paths
2. Interference Mechanisms: Hypotheses interact and interfere
3. Context-Based Collapse: Attention acts as measurement operator
4. Entangled Representations: Correlated token embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math


class SuperpositionEmbedding(nn.Module):
    """
    Embeds tokens into a superposition of N parallel hypothesis spaces.
    Each token exists simultaneously in multiple interpretations.
    """
    def __init__(self, vocab_size: int, d_model: int, n_hypotheses: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses

        # Each hypothesis gets its own embedding space
        self.hypothesis_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(n_hypotheses)
        ])

        # Phase parameters for quantum-inspired interference
        self.phases = nn.Parameter(torch.randn(n_hypotheses, d_model) * 2 * math.pi)

        # Amplitude (probability) for each hypothesis
        self.amplitudes = nn.Parameter(torch.ones(n_hypotheses) / math.sqrt(n_hypotheses))

    def forward(self, x):
        """
        Returns superposed embeddings: (batch, seq_len, n_hypotheses, d_model)
        """
        batch_size, seq_len = x.shape

        # Get embeddings from each hypothesis space
        hypothesis_embeds = []
        for i, emb_layer in enumerate(self.hypothesis_embeddings):
            h_emb = emb_layer(x)  # (batch, seq_len, d_model)

            # Apply phase rotation (quantum-inspired)
            phase = self.phases[i].unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            h_emb_complex = h_emb * torch.exp(1j * phase)

            # Apply amplitude
            h_emb_complex = h_emb_complex * self.amplitudes[i]

            hypothesis_embeds.append(h_emb_complex)

        # Stack hypotheses: (batch, seq_len, n_hypotheses, d_model)
        superposition = torch.stack([h.real for h in hypothesis_embeds], dim=2)

        return superposition


class InterferenceLayer(nn.Module):
    """
    Allows hypotheses to interfere with each other, creating constructive
    and destructive interference patterns based on learned relationships.
    """
    def __init__(self, d_model: int, n_hypotheses: int):
        super().__init__()
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses

        # Interference matrix: how hypotheses interact
        self.interference_weights = nn.Parameter(
            torch.eye(n_hypotheses) + 0.1 * torch.randn(n_hypotheses, n_hypotheses)
        )

        # Learnable interference strength
        self.interference_strength = nn.Parameter(torch.tensor(0.3))

    def forward(self, superposition):
        """
        Input: (batch, seq_len, n_hypotheses, d_model)
        Output: (batch, seq_len, n_hypotheses, d_model) with interference applied
        """
        batch_size, seq_len, n_hyp, d_model = superposition.shape

        # Normalize interference weights to maintain stability
        interference = F.softmax(self.interference_weights, dim=-1)

        # Reshape for matrix multiplication
        # (batch, seq_len, n_hyp, d_model) -> (batch*seq_len, n_hyp, d_model)
        x = superposition.reshape(-1, n_hyp, d_model)

        # Apply interference: each hypothesis influenced by others
        # (batch*seq_len, n_hyp, d_model) @ (n_hyp, n_hyp) -> (batch*seq_len, n_hyp, d_model)
        interfered = torch.einsum('bnd,hj->bjd', x, interference)

        # Blend original and interfered states
        output = (1 - self.interference_strength) * x + self.interference_strength * interfered

        # Reshape back
        output = output.reshape(batch_size, seq_len, n_hyp, d_model)

        return output


class QuantumAttention(nn.Module):
    """
    Attention mechanism that acts as a measurement operator,
    collapsing the superposition based on context.
    """
    def __init__(self, d_model: int, n_hypotheses: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0

        # Separate Q, K, V for each hypothesis
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Collapse weights: determines which hypothesis to favor
        self.collapse_net = nn.Sequential(
            nn.Linear(d_model, n_hypotheses),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, superposition):
        """
        Input: (batch, seq_len, n_hypotheses, d_model)
        Output: (batch, seq_len, d_model) - collapsed state
        """
        batch_size, seq_len, n_hyp, d_model = superposition.shape

        # Process each hypothesis through attention
        attended_hypotheses = []
        collapse_weights_list = []

        for h in range(n_hyp):
            h_state = superposition[:, :, h, :]  # (batch, seq_len, d_model)

            # Multi-head attention
            q = self.q_proj(h_state).reshape(batch_size, seq_len, self.n_heads, self.d_head)
            k = self.k_proj(h_state).reshape(batch_size, seq_len, self.n_heads, self.d_head)
            v = self.v_proj(h_state).reshape(batch_size, seq_len, self.n_heads, self.d_head)

            # Transpose for attention: (batch, n_heads, seq_len, d_head)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn_weights = F.softmax(scores, dim=-1)
            attended = torch.matmul(attn_weights, v)

            # Reshape back: (batch, seq_len, d_model)
            attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
            attended = self.out_proj(attended)

            attended_hypotheses.append(attended)

            # Calculate collapse weights for this hypothesis
            collapse_weights = self.collapse_net(attended)  # (batch, seq_len, n_hyp)
            collapse_weights_list.append(collapse_weights)

        # Stack attended hypotheses
        attended_stack = torch.stack(attended_hypotheses, dim=2)  # (batch, seq_len, n_hyp, d_model)

        # Average collapse weights across hypotheses
        collapse_weights = torch.stack(collapse_weights_list, dim=2).mean(dim=2)  # (batch, seq_len, n_hyp)

        # Collapse superposition using weighted sum
        collapsed = torch.einsum('bshd,bsh->bsd', attended_stack, collapse_weights)

        return collapsed, collapse_weights


class SuperpositionTransformerBlock(nn.Module):
    """
    Complete transformer block operating in superposition space.
    """
    def __init__(self, d_model: int, n_hypotheses: int, n_heads: int = 4, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.d_ff = d_ff or 4 * d_model

        self.interference = InterferenceLayer(d_model, n_hypotheses)
        self.quantum_attn = QuantumAttention(d_model, n_hypotheses, n_heads)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, d_model)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Expansion back to superposition
        self.expand = nn.Linear(d_model, d_model * n_hypotheses)

    def forward(self, superposition):
        """
        Input: (batch, seq_len, n_hypotheses, d_model)
        Output: (batch, seq_len, n_hypotheses, d_model), collapse_weights
        """
        # Apply interference between hypotheses
        interfered = self.interference(superposition)

        # Collapse through attention
        collapsed, collapse_weights = self.quantum_attn(interfered)

        # Feed-forward on collapsed state
        ff_out = self.ff(self.norm1(collapsed))
        collapsed = collapsed + ff_out
        collapsed = self.norm2(collapsed)

        # Expand back to superposition
        batch_size, seq_len, d_model = collapsed.shape
        expanded = self.expand(collapsed)  # (batch, seq_len, d_model * n_hyp)
        expanded = expanded.reshape(batch_size, seq_len, self.n_hypotheses, d_model)

        # Residual connection in superposition space
        output = interfered + expanded

        return output, collapse_weights


class QuantumSuperpositionTransformer(nn.Module):
    """
    Complete Quantum-Inspired Superposition Transformer.
    Processes information through parallel hypothesis spaces.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_hypotheses: int = 4,
        n_layers: int = 3,
        n_heads: int = 4,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.n_layers = n_layers

        # Superposition embedding
        self.embedding = SuperpositionEmbedding(vocab_size, d_model, n_hypotheses)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SuperpositionTransformerBlock(d_model, n_hypotheses, n_heads)
            for _ in range(n_layers)
        ])

        # Final collapse and output
        self.final_collapse = QuantumAttention(d_model, n_hypotheses, n_heads)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Input: (batch, seq_len) - token indices
        Output: (batch, seq_len, vocab_size) - logits
        """
        batch_size, seq_len = x.shape

        # Create superposition embeddings
        superposition = self.embedding(x)  # (batch, seq_len, n_hyp, d_model)

        # Add positional encoding to each hypothesis
        pos_enc = self.pos_encoding[:, :seq_len, :].unsqueeze(2)  # (1, seq_len, 1, d_model)
        superposition = superposition + pos_enc

        # Track collapse weights through layers
        all_collapse_weights = []

        # Process through transformer blocks
        for block in self.blocks:
            superposition, collapse_weights = block(superposition)
            all_collapse_weights.append(collapse_weights)

        # Final collapse
        collapsed, final_collapse_weights = self.final_collapse(superposition)
        all_collapse_weights.append(final_collapse_weights)

        # Output projection
        logits = self.output_proj(collapsed)

        return logits, all_collapse_weights


def visualize_superposition_dynamics(model, input_seq, vocab):
    """
    Visualize how the superposition evolves and collapses through the network.
    """
    model.eval()
    with torch.no_grad():
        logits, collapse_weights = model(input_seq)

    # Create visualization
    n_layers = len(collapse_weights)
    fig, axes = plt.subplots(n_layers, 1, figsize=(14, 3 * n_layers))

    if n_layers == 1:
        axes = [axes]

    for layer_idx, (ax, weights) in enumerate(zip(axes, collapse_weights)):
        # weights: (batch, seq_len, n_hypotheses)
        weights_np = weights[0].cpu().numpy()  # Take first batch item

        # Plot heatmap
        im = ax.imshow(weights_np.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Hypothesis')
        ax.set_title(f'Layer {layer_idx + 1}: Hypothesis Collapse Weights')
        plt.colorbar(im, ax=ax)

        # Add token labels if available
        if vocab and len(vocab) < 50:
            seq_tokens = [vocab.get(idx.item(), f"T{idx.item()}") for idx in input_seq[0]]
            ax.set_xticks(range(len(seq_tokens)))
            ax.set_xticklabels(seq_tokens, rotation=45, ha='right')

    plt.tight_layout()
    return fig


def run_interference_experiment():
    """
    Experiment: Show how hypotheses interfere with each other.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Hypothesis Interference Patterns")
    print("="*60)

    # Create a simple vocabulary
    vocab_size = 20
    vocab = {i: f"W{i}" for i in range(vocab_size)}

    # Initialize model
    model = QuantumSuperpositionTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_hypotheses=4,
        n_layers=3,
        n_heads=4
    )

    # Create test sequence
    seq_len = 16
    test_seq = torch.randint(0, vocab_size, (1, seq_len))

    print(f"\nInput sequence: {[vocab[idx.item()] for idx in test_seq[0]]}")

    # Forward pass
    logits, collapse_weights = model(test_seq)

    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Processing through {model.n_layers} layers with {model.n_hypotheses} parallel hypotheses")

    # Analyze collapse patterns
    print("\n--- Hypothesis Collapse Analysis ---")
    for layer_idx, weights in enumerate(collapse_weights):
        weights_np = weights[0].cpu().detach().numpy()
        entropy = -np.sum(weights_np * np.log(weights_np + 1e-10), axis=-1).mean()

        dominant_hyp = weights_np.argmax(axis=-1)
        unique, counts = np.unique(dominant_hyp, return_counts=True)

        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Average entropy: {entropy:.3f} (higher = more uncertainty)")
        print(f"  Dominant hypothesis distribution: {dict(zip(unique, counts))}")
        print(f"  Mean weights: {weights_np.mean(axis=(0,1))}")

    # Visualize
    fig = visualize_superposition_dynamics(model, test_seq, vocab)
    plt.savefig('/home/burny/projects/open-ended-vibecoded-ai/interference_patterns.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: interference_patterns.png")

    return model, test_seq, vocab


def run_training_experiment():
    """
    Experiment: Train on a simple sequence prediction task.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Training on Sequence Prediction")
    print("="*60)

    # Create synthetic task: predict next token in arithmetic sequences
    vocab_size = 20
    model = QuantumSuperpositionTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_hypotheses=4,
        n_layers=2,
        n_heads=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate training data: simple repeating patterns
    def generate_sequence_batch(batch_size=32, seq_len=16):
        sequences = []
        for _ in range(batch_size):
            # Random pattern
            pattern_len = np.random.randint(2, 5)
            pattern = torch.randint(0, vocab_size, (pattern_len,))

            # Repeat pattern
            seq = pattern.repeat((seq_len // pattern_len) + 1)[:seq_len]
            sequences.append(seq)

        return torch.stack(sequences)

    # Training loop
    print("\nTraining model to learn repeating patterns...")
    losses = []

    for epoch in range(100):
        batch = generate_sequence_batch(batch_size=32, seq_len=16)

        # Input: all but last token, Target: all but first token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, collapse_weights = model(inputs)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress: Quantum Superposition Transformer')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/burny/projects/open-ended-vibecoded-ai/training_curve.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curve saved to: training_curve.png")

    # Test generalization
    print("\n--- Testing Generalization ---")
    test_batch = generate_sequence_batch(batch_size=3, seq_len=16)

    model.eval()
    with torch.no_grad():
        test_inputs = test_batch[:, :-1]
        test_targets = test_batch[:, 1:]
        logits, _ = model(test_inputs)
        predictions = logits.argmax(dim=-1)

        accuracy = (predictions == test_targets).float().mean().item()
        print(f"Next-token prediction accuracy: {accuracy*100:.2f}%")

        # Show example
        print("\nExample prediction:")
        print(f"Input:  {test_inputs[0].tolist()}")
        print(f"Target: {test_targets[0].tolist()}")
        print(f"Pred:   {predictions[0].tolist()}")

    return model, losses


def run_hypothesis_divergence_experiment(model, test_seq, vocab):
    """
    Experiment: Analyze how hypotheses diverge and converge.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Hypothesis Divergence Analysis")
    print("="*60)

    model.eval()

    # Hook to extract hypothesis states
    hypothesis_states = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            superposition = output[0]
        else:
            superposition = output
        hypothesis_states.append(superposition.detach())

    # Register hooks
    hooks = []
    for block in model.blocks:
        hook = block.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        logits, collapse_weights = model(test_seq)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze hypothesis divergence
    print("\n--- Hypothesis State Analysis ---")

    fig, axes = plt.subplots(len(hypothesis_states), 1, figsize=(12, 3 * len(hypothesis_states)))
    if len(hypothesis_states) == 1:
        axes = [axes]

    for layer_idx, (ax, h_state) in enumerate(zip(axes, hypothesis_states)):
        # h_state: (batch, seq_len, n_hypotheses, d_model)
        h_state_np = h_state[0].cpu().numpy()  # (seq_len, n_hypotheses, d_model)

        # Calculate pairwise distances between hypotheses
        n_hyp = h_state_np.shape[1]
        distances = np.zeros((n_hyp, n_hyp))

        for i in range(n_hyp):
            for j in range(n_hyp):
                # Average L2 distance across sequence
                dist = np.linalg.norm(h_state_np[:, i, :] - h_state_np[:, j, :], axis=-1).mean()
                distances[i, j] = dist

        # Plot distance matrix
        im = ax.imshow(distances, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Hypothesis')
        ax.set_ylabel('Hypothesis')
        ax.set_title(f'Layer {layer_idx + 1}: Hypothesis Divergence (L2 Distance)')
        plt.colorbar(im, ax=ax)

        # Print statistics
        off_diagonal = distances[np.triu_indices_from(distances, k=1)]
        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Mean hypothesis distance: {off_diagonal.mean():.3f}")
        print(f"  Std hypothesis distance: {off_diagonal.std():.3f}")

    plt.tight_layout()
    plt.savefig('/home/burny/projects/open-ended-vibecoded-ai/hypothesis_divergence.png', dpi=150, bbox_inches='tight')
    print(f"\nDivergence analysis saved to: hypothesis_divergence.png")

    plt.close('all')


def main():
    """
    Run all experiments for the Quantum-Inspired Superposition Transformer.
    """
    print("\n" + "="*60)
    print("QUANTUM-INSPIRED SUPERPOSITION TRANSFORMER")
    print("="*60)
    print("\nA novel neural architecture that:")
    print("  • Maintains multiple parallel hypothesis spaces")
    print("  • Uses interference patterns between hypotheses")
    print("  • Collapses superposition through context-based attention")
    print("  • Demonstrates quantum-inspired information processing")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run experiments
    model, test_seq, vocab = run_interference_experiment()
    run_hypothesis_divergence_experiment(model, test_seq, vocab)
    trained_model, losses = run_training_experiment()

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print("\nGenerated outputs:")
    print("  • interference_patterns.png - Shows how hypotheses collapse through layers")
    print("  • hypothesis_divergence.png - Shows how hypotheses diverge in representation space")
    print("  • training_curve.png - Shows learning dynamics")

    print("\n" + "="*60)
    print("ARCHITECTURAL INNOVATIONS SUMMARY")
    print("="*60)
    print("""
1. SUPERPOSITION EMBEDDINGS
   - Each token exists in multiple hypothesis spaces simultaneously
   - Phase-rotated embeddings create interference patterns

2. INTERFERENCE LAYER
   - Hypotheses interact through learned interference weights
   - Constructive/destructive interference shapes representations

3. QUANTUM ATTENTION
   - Attention acts as a 'measurement' operator
   - Collapses superposition probabilistically based on context
   - Different contexts favor different hypotheses

4. HYPOTHESIS DYNAMICS
   - Information flows through parallel reality branches
   - Branches interfere and influence each other
   - Final output emerges from collapsed superposition

This architecture explores what happens when we explicitly model
multiple competing interpretations of information, allowing them
to coexist and interact before collapsing to a final answer.
    """)


if __name__ == "__main__":
    main()
