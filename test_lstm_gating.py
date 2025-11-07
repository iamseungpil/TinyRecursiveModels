"""
Test script for comparing baseline TRM with LSTM-gated TRM.

This script:
1. Creates both baseline and LSTM-gated models
2. Runs forward passes on sample data
3. Compares outputs and measures performance
4. Logs LSTM gate activation statistics
"""

import torch
import time
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

def create_sample_batch(batch_size, seq_len, vocab_size, num_puzzle_ids):
    """Create a sample batch for testing."""
    return {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, num_puzzle_ids, (batch_size,)),
    }

def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_baseline_model():
    """Test baseline TRM without LSTM gating."""
    print("="*80)
    print("Testing Baseline TRM (no LSTM gating)")
    print("="*80)

    config = {
        'batch_size': 4,
        'seq_len': 30,
        'puzzle_emb_ndim': 128,
        'num_puzzle_identifiers': 100,
        'vocab_size': 256,
        'H_cycles': 3,
        'L_cycles': 4,
        'H_layers': 0,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 8,
        'halt_exploration_prob': 0.1,
        'use_lstm_gating': False,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.eval()

    print(f"✓ Model created")
    print(f"  Parameters: {count_parameters(model):,}")

    # Create sample batch
    batch = create_sample_batch(config['batch_size'], config['seq_len'],
                                config['vocab_size'], config['num_puzzle_identifiers'])

    # Initialize carry
    carry = model.initial_carry(batch)
    print(f"✓ Initial carry created")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        new_carry, outputs = model(carry, batch)
    elapsed = time.time() - start_time

    print(f"✓ Forward pass completed in {elapsed*1000:.2f}ms")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Q halt logits mean: {outputs['q_halt_logits'].mean().item():.4f}")
    print(f"  Q halt logits std: {outputs['q_halt_logits'].std().item():.4f}")

    return model, new_carry, outputs

def test_lstm_gated_model():
    """Test LSTM-gated TRM."""
    print("\n" + "="*80)
    print("Testing LSTM-Gated TRM")
    print("="*80)

    config = {
        'batch_size': 4,
        'seq_len': 30,
        'puzzle_emb_ndim': 128,
        'num_puzzle_identifiers': 100,
        'vocab_size': 256,
        'H_cycles': 3,
        'L_cycles': 4,
        'H_layers': 0,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 8,
        'halt_exploration_prob': 0.1,
        'use_lstm_gating': True,
        'lstm_init_forget_bias': 1.0,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.eval()

    print(f"✓ Model created with LSTM gating")
    print(f"  Parameters: {count_parameters(model):,}")

    # Create sample batch
    batch = create_sample_batch(config['batch_size'], config['seq_len'],
                                config['vocab_size'], config['num_puzzle_identifiers'])

    # Initialize carry
    carry = model.initial_carry(batch)
    print(f"✓ Initial carry created (with cell state)")
    print(f"  Cell state shape: {carry.inner_carry.c_H.shape if carry.inner_carry.c_H is not None else 'None'}")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        new_carry, outputs = model(carry, batch)
    elapsed = time.time() - start_time

    print(f"✓ Forward pass completed in {elapsed*1000:.2f}ms")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Q halt logits mean: {outputs['q_halt_logits'].mean().item():.4f}")
    print(f"  Q halt logits std: {outputs['q_halt_logits'].std().item():.4f}")

    # Get gate statistics
    print("\n" + "-"*80)
    print("LSTM Gate Statistics:")
    print("-"*80)

    lstm_context = model.inner.lstm_context

    # Get gate stats by running through the LSTM module
    # We'll use the final states from the carry
    gate_stats = lstm_context.get_gate_statistics(
        new_carry.inner_carry.c_H,
        new_carry.inner_carry.z_H,
        new_carry.inner_carry.z_L
    )

    for key, value in gate_stats.items():
        print(f"  {key}: {value:.4f}")

    return model, new_carry, outputs, gate_stats

def compare_models():
    """Compare baseline and LSTM-gated models."""
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)

    baseline_model, baseline_carry, baseline_outputs = test_baseline_model()
    lstm_model, lstm_carry, lstm_outputs, gate_stats = test_lstm_gated_model()

    baseline_params = count_parameters(baseline_model)
    lstm_params = count_parameters(lstm_model)
    param_overhead = lstm_params - baseline_params
    param_overhead_pct = (param_overhead / baseline_params) * 100

    print(f"\nParameter Comparison:")
    print(f"  Baseline: {baseline_params:,}")
    print(f"  LSTM-gated: {lstm_params:,}")
    print(f"  Overhead: {param_overhead:,} ({param_overhead_pct:.2f}%)")

    print(f"\nOutput Comparison:")
    print(f"  Baseline Q halt mean: {baseline_outputs['q_halt_logits'].mean().item():.4f}")
    print(f"  LSTM-gated Q halt mean: {lstm_outputs['q_halt_logits'].mean().item():.4f}")

    print("\n" + "="*80)
    print("✅ All tests completed successfully!")
    print("="*80)

    return {
        'baseline': {'model': baseline_model, 'carry': baseline_carry, 'outputs': baseline_outputs},
        'lstm': {'model': lstm_model, 'carry': lstm_carry, 'outputs': lstm_outputs, 'gate_stats': gate_stats},
    }

if __name__ == "__main__":
    results = compare_models()

    print("\n" + "="*80)
    print("LSTM Gate Analysis Summary")
    print("="*80)
    stats = results['lstm']['gate_stats']

    print("\nGate Activation Patterns:")
    print(f"  Forget gate: {stats['forget_gate_mean']:.4f} ± {stats['forget_gate_std']:.4f}")
    print(f"    → Average retention: {stats['forget_gate_mean']*100:.1f}%")
    print(f"  Input gate: {stats['input_gate_mean']:.4f} ± {stats['input_gate_std']:.4f}")
    print(f"    → Average new info intake: {stats['input_gate_mean']*100:.1f}%")
    print(f"  Output gate: {stats['output_gate_mean']:.4f} ± {stats['output_gate_std']:.4f}")
    print(f"    → Average information exposure: {stats['output_gate_mean']*100:.1f}%")

    print(f"\nContext Statistics:")
    print(f"  Cell state norm: {stats['cell_state_norm']:.4f}")
    print(f"  Context norm: {stats['context_norm']:.4f}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Train both models on ARC dataset")
    print("2. Compare convergence speed and final performance")
    print("3. Analyze gate patterns on different puzzle types")
    print("4. Visualize context evolution across H-cycles")
    print("="*80)
