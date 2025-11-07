"""
Direct comparison: POC manual H-cycling vs ACT wrapper

This script runs the SAME model on the SAME puzzle using both approaches
to identify why they give different results.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/ubuntu/TinyRecursiveModels')

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def load_model():
    """Load model from checkpoint."""
    checkpoint_path = '/data/trm/checkpoints/pretrain_att_arc1concept_4/step_518071'
    device = 'cuda'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]

    # Clean prefixes
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.model.', '').replace('_orig_mod.', '').replace('model.', '')
        cleaned[k] = v

    config = {
        "batch_size": 1,
        "seq_len": 900,
        "vocab_size": 12,
        "num_puzzle_identifiers": 876406,
        "puzzle_emb_ndim": 512,
        "puzzle_emb_len": 16,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,
        "pos_encodings": "rope",
    }

    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()

    return model, config

def load_puzzle_0():
    """Load puzzle 0 from test set."""
    data_path = '/data/arc1concept-aug-1000'
    test_dir = os.path.join(data_path, 'test')

    inputs = np.load(os.path.join(test_dir, 'all__inputs.npy'), mmap_mode='r')
    labels = np.load(os.path.join(test_dir, 'all__labels.npy'), mmap_mode='r')
    puzzle_ids = np.load(os.path.join(test_dir, 'all__puzzle_identifiers.npy'))
    puzzle_indices = np.load(os.path.join(test_dir, 'all__puzzle_indices.npy'))

    start = puzzle_indices[0]
    end = puzzle_indices[1]

    batch = {
        'inputs': torch.from_numpy(inputs[start:end].astype(np.int64)).cuda(),
        'labels': torch.from_numpy(labels[start:end].astype(np.int64)).cuda(),
        'puzzle_identifiers': torch.full((end-start,), puzzle_ids[0], dtype=torch.long, device='cuda')
    }

    return batch

def poc_inference(model, batch, num_h_steps=3):
    """POC-style: manual H-cycling."""
    print("\n=== POC-style inference (manual H-cycling) ===")

    inner = model.inner
    batch_size = batch['inputs'].shape[0]

    # Initialize
    z_H = inner.H_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, 900 + 16, -1)
    z_L = inner.L_init.unsqueeze(0).expand(batch_size, -1).unsqueeze(1).expand(-1, 900 + 16, -1)

    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])

    labels_np = batch['labels'].cpu().numpy()

    with torch.no_grad():
        for h in range(num_h_steps):
            # L-cycles
            for l in range(model.config.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)

            # H-cycle
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Output
            logits = inner.lm_head(z_H)[:, 16:]
            preds = logits.argmax(dim=-1).cpu().numpy()
            acc = (preds == labels_np).mean()

            print(f"  H-step {h}: accuracy = {acc:.1%}")

    return acc

def act_inference(model, batch, num_act_steps=3):
    """ACT-style: using the wrapper."""
    print("\n=== ACT-style inference (using wrapper) ===")

    carry = model.initial_carry(batch)
    labels_np = batch['labels'].cpu().numpy()

    with torch.no_grad():
        for step in range(num_act_steps):
            carry, outputs = model(carry, batch)

            logits = outputs['logits']
            preds = logits.argmax(dim=-1).cpu().numpy()
            acc = (preds == labels_np).mean()

            print(f"  ACT-step {step} (= {(step+1)*3} H-cycles): accuracy = {acc:.1%}")

    return acc

def main():
    print("="*80)
    print("Comparing POC vs ACT inference on Puzzle 0")
    print("="*80)

    model, config = load_model()
    batch = load_puzzle_0()

    print(f"\nModel config:")
    print(f"  H_cycles: {config['H_cycles']}")
    print(f"  L_cycles: {config['L_cycles']}")
    print(f"  halt_max_steps: {config['halt_max_steps']}")

    print(f"\nBatch:")
    print(f"  inputs shape: {batch['inputs'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    # Test both
    poc_acc = poc_inference(model, batch, num_h_steps=3)
    act_acc = act_inference(model, batch, num_act_steps=3)

    print(f"\n{'='*80}")
    print(f"Final Results:")
    print(f"  POC (3 H-steps): {poc_acc:.1%}")
    print(f"  ACT (3 ACT-steps = 9 H-cycles): {act_acc:.1%}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
