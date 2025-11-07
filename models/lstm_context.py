"""
LSTM-Style Gating for TRM Context Accumulation

This module implements LSTM-style gating to enable selective information flow
in the TRM's H-cycle context accumulation. Instead of unconditionally accumulating
all information, LSTM gates allow the model to:
- Forget outdated or irrelevant information (forget gate)
- Selectively add new information (input gate)
- Control what information is exposed (output gate)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMStyleContext(nn.Module):
    """
    LSTM-style gating for selective context accumulation.

    Given previous context (z_H) and new thought (z_L output), this module
    computes gated updates to maintain a cell state (long-term memory) and
    hidden state (working context).

    Args:
        hidden_dim: Dimension of the hidden state and cell state
        init_forget_bias: Initial bias for forget gate (default 1.0 for remembering)
        dtype: Data type for the module (default: torch.float32)
    """

    def __init__(self, hidden_dim: int, init_forget_bias: float = 1.0, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Gates: each takes concatenation of [context, new_thought]
        # Input dimension is 2*hidden_dim, output is hidden_dim
        self.forget_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True, dtype=dtype)
        self.input_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True, dtype=dtype)
        self.output_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True, dtype=dtype)
        self.cell_update = nn.Linear(hidden_dim * 2, hidden_dim, bias=True, dtype=dtype)

        # Initialize forget gate bias to encourage remembering initially
        # This is a standard LSTM trick to avoid vanishing gradients early in training
        nn.init.constant_(self.forget_gate.bias, init_forget_bias)

    def forward(
        self,
        cell_state: torch.Tensor,
        context_t: torch.Tensor,
        new_thought_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LSTM-style gating.

        Args:
            cell_state: Previous cell state (B, L, D) - long-term memory
            context_t: Previous context/hidden state (B, L, D) - working memory
            new_thought_t: New information from L-cycle (B, L, D)

        Returns:
            h_t: New hidden state (B, L, D)
            c_t: New cell state (B, L, D)
        """
        # Concatenate previous context and new thought for gate computation
        # Shape: (B, L, 2*D)
        combined = torch.cat([context_t, new_thought_t], dim=-1)

        # Compute gates (all in range [0, 1] after sigmoid)
        f_t = torch.sigmoid(self.forget_gate(combined))    # Forget gate
        i_t = torch.sigmoid(self.input_gate(combined))     # Input gate
        o_t = torch.sigmoid(self.output_gate(combined))    # Output gate

        # Compute candidate cell state (in range [-1, 1] after tanh)
        c_tilde = torch.tanh(self.cell_update(combined))

        # Update cell state:
        # - Forget some of the old information (f_t * cell_state)
        # - Add some of the new information (i_t * c_tilde)
        c_t = f_t * cell_state + i_t * c_tilde

        # Compute hidden state (working context):
        # - Take current cell state and apply output gating
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def get_gate_statistics(
        self,
        cell_state: torch.Tensor,
        context_t: torch.Tensor,
        new_thought_t: torch.Tensor
    ) -> dict:
        """
        Compute gate activation statistics for logging/analysis.

        Args:
            cell_state: Previous cell state (B, L, D)
            context_t: Previous context (B, L, D)
            new_thought_t: New thought (B, L, D)

        Returns:
            Dictionary with gate statistics (mean, std for each gate)
        """
        with torch.no_grad():
            combined = torch.cat([context_t, new_thought_t], dim=-1)

            f_t = torch.sigmoid(self.forget_gate(combined))
            i_t = torch.sigmoid(self.input_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))
            c_tilde = torch.tanh(self.cell_update(combined))
            c_t = f_t * cell_state + i_t * c_tilde

            return {
                'forget_gate_mean': f_t.mean().item(),
                'forget_gate_std': f_t.std().item(),
                'input_gate_mean': i_t.mean().item(),
                'input_gate_std': i_t.std().item(),
                'output_gate_mean': o_t.mean().item(),
                'output_gate_std': o_t.std().item(),
                'cell_state_norm': c_t.norm(dim=-1).mean().item(),
                'context_norm': context_t.norm(dim=-1).mean().item(),
            }
