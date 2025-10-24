"""
Test TRM Hybrid Components without loading full LLaMA model
"""
import torch
import sys
sys.path.append('/home/ubuntu/TinyRecursiveModels/gpt-integration')
sys.path.append('/home/ubuntu/TinyRecursiveModels')

from models.trm_hybrid import HybridTRM_Inner, HybridTRMConfig, HybridTRMInnerCarry

print("Testing TRM Hybrid Inner Model (without LLaMA)...")

# Config
config = HybridTRMConfig(
    batch_size=2,
    seq_len=900,
    vocab_size=12,
    H_cycles=2,
    L_cycles=1,
    H_layers=0,
    L_layers=4,
    hidden_size=512,
    text_hidden_size=4096,
    expansion=2.0,
    num_heads=8,
    pos_encodings='rope',
    num_puzzle_identifiers=1,
    puzzle_emb_ndim=0
)

# Create inner model
inner_model = HybridTRM_Inner(config).cuda()
print(f"✓ Inner model created")

# Create test inputs
batch_size = 2
z_init = torch.randn(batch_size, 4096).cuda()  # Simulated LLaMA output
problem_grid = torch.randint(0, 12, (batch_size, 900)).cuda()

# Create initial carry
carry = inner_model.empty_carry(batch_size).z_H.cuda()
carry_full = HybridTRMInnerCarry(
    z_H=torch.randn(batch_size, 900, 512).cuda(),
    z_L=torch.randn(batch_size, 900, 512).cuda()
)

# Reset carry
reset_flag = torch.ones(batch_size, dtype=torch.bool).cuda()
carry_reset = inner_model.reset_carry(reset_flag, carry_full)
print(f"✓ Carry reset successful")

# Forward pass
new_carry, logits = inner_model(carry_reset, z_init, problem_grid)

print(f"✓ Forward pass successful")
print(f"  Logits shape: {logits.shape}")  # Should be [2, 900, 12]
print(f"  Carry z_H shape: {new_carry.z_H.shape}")  # Should be [2, 900, 512]
print(f"  Carry z_L shape: {new_carry.z_L.shape}")  # Should be [2, 900, 512]

# Test backward pass
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, 12),
    problem_grid.view(-1)
)
loss.backward()
print(f"✓ Backward pass successful")
print(f"  Loss: {loss.item():.4f}")

print("\n✅ TRM Hybrid Inner Model test passed!")
