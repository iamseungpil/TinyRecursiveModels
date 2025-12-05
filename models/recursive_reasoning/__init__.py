"""
Recursive reasoning models for TinyRecursiveModels.

This module provides recursive reasoning model implementations:
- TRM (Tiny Recursive Model) variants
- HRM (Hierarchical Recursive Model)
- Encoder components for goal conditioning
"""

# Note: Model classes are imported on demand to avoid circular imports
# Use explicit imports like:
#   from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
#   from models.recursive_reasoning.encoder import GridEncoder
