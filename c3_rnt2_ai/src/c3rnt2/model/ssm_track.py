from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SSMState:
    hidden: torch.Tensor


class SSMTrack(nn.Module):
    """Minimal recurrent SSM block with O(n) scan."""

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.gru = nn.GRU(hidden_size, state_size, batch_first=True)
        self.out_proj = nn.Linear(state_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_in = x.contiguous()
        if orig_dtype == torch.bfloat16:
            x_in = x_in.float()
        try:
            out, _ = self.gru(x_in)
        except RuntimeError as exc:
            if "CUDNN_STATUS_NOT_SUPPORTED" in str(exc):
                prev = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                out, _ = self.gru(x_in)
                torch.backends.cudnn.enabled = prev
            else:
                raise
        if orig_dtype == torch.bfloat16:
            out = out.to(orig_dtype)
        return self.out_proj(out)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> SSMState:
        hidden = torch.zeros(1, batch, self.state_size, device=device, dtype=dtype)
        return SSMState(hidden=hidden)

    def step(self, x: torch.Tensor, state: SSMState) -> tuple[torch.Tensor, SSMState]:
        # x: [B, H]
        orig_dtype = x.dtype
        x_in = x.unsqueeze(1)
        h_in = state.hidden
        if orig_dtype == torch.bfloat16:
            x_in = x_in.float()
            h_in = h_in.float()
        try:
            out, hidden = self.gru(x_in, h_in)
        except RuntimeError as exc:
            if "CUDNN_STATUS_NOT_SUPPORTED" in str(exc):
                prev = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                out, hidden = self.gru(x_in, h_in)
                torch.backends.cudnn.enabled = prev
            else:
                raise
        if orig_dtype == torch.bfloat16:
            out = out.to(orig_dtype)
            hidden = hidden.to(orig_dtype)
        if not self.training:
            hidden = hidden.detach()
        return self.out_proj(out.squeeze(1)), SSMState(hidden=hidden)

    def step_block(self, x: torch.Tensor, state: SSMState) -> tuple[torch.Tensor, SSMState]:
        # x: [B, K, H]
        if x.dim() != 3:
            raise ValueError("SSMTrack.step_block expects [B, K, H]")
        orig_dtype = x.dtype
        x_in = x
        h_in = state.hidden
        if orig_dtype == torch.bfloat16:
            x_in = x_in.float()
            h_in = h_in.float()
        try:
            out, hidden = self.gru(x_in, h_in)
        except RuntimeError as exc:
            if "CUDNN_STATUS_NOT_SUPPORTED" in str(exc):
                prev = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                out, hidden = self.gru(x_in, h_in)
                torch.backends.cudnn.enabled = prev
            else:
                raise
        if orig_dtype == torch.bfloat16:
            out = out.to(orig_dtype)
            hidden = hidden.to(orig_dtype)
        if not self.training:
            hidden = hidden.detach()
        return self.out_proj(out), SSMState(hidden=hidden)
