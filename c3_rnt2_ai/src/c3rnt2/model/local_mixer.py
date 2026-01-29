from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LocalMixerState:
    buffer: torch.Tensor


class LocalMixer(nn.Module):
    """Depthwise conv local mixer for syntax patterns."""

    def __init__(self, hidden_size: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        y = x.transpose(1, 2)
        if self.kernel_size > 1:
            y = F.pad(y, (self.kernel_size - 1, 0))
        y = self.conv(y)
        return y.transpose(1, 2)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> LocalMixerState:
        if self.kernel_size <= 1:
            buf = torch.zeros(batch, 0, self.hidden_size, device=device, dtype=dtype)
        else:
            buf = torch.zeros(batch, self.kernel_size - 1, self.hidden_size, device=device, dtype=dtype)
        return LocalMixerState(buffer=buf)

    def step(self, x: torch.Tensor, state: LocalMixerState) -> tuple[torch.Tensor, LocalMixerState]:
        # x: [B, H]
        if self.kernel_size <= 1:
            window = x.unsqueeze(2)
        else:
            window = torch.cat([state.buffer, x.unsqueeze(1)], dim=1)
        if self.kernel_size > 1:
            window_bhk = window.transpose(1, 2)
        else:
            window_bhk = window
        weight = self.conv.weight.squeeze(1)
        out = (window_bhk * weight.unsqueeze(0)).sum(dim=2)
        if self.conv.bias is not None:
            out = out + self.conv.bias
        if self.kernel_size > 1:
            new_buf = window[:, 1:, :]
            if not self.training:
                new_buf = new_buf.detach()
            state = LocalMixerState(buffer=new_buf)
        return out, state
