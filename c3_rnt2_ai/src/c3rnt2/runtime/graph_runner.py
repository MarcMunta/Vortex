from __future__ import annotations

from typing import List, Tuple

import torch

from ..model.local_mixer import LocalMixerState
from ..model.ssm_track import SSMState
from ..model.vblock import VBlockState


def _clone_state(state: List[VBlockState]) -> List[VBlockState]:
    cloned: List[VBlockState] = []
    for item in state:
        local = LocalMixerState(buffer=item.local.buffer.detach().clone())
        ssm = SSMState(hidden=item.ssm.hidden.detach().clone())
        prev = item.prev.detach().clone() if item.prev is not None else None
        cloned.append(VBlockState(local=local, ssm=ssm, prev=prev))
    return cloned


def _copy_state(dst: List[VBlockState], src: List[VBlockState]) -> None:
    for d, s in zip(dst, src):
        d.local.buffer.copy_(s.local.buffer)
        d.ssm.hidden.copy_(s.ssm.hidden)
        if d.prev is not None and s.prev is not None:
            d.prev.copy_(s.prev)


class GraphRunner:
    def __init__(
        self,
        model,
        state_template: List[VBlockState],
        token_shape: Tuple[int, int],
        return_mtp: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graphs require CUDA")
        self.model = model
        self.device = model.device
        self.token_static = torch.zeros(token_shape, dtype=torch.long, device=self.device)
        self.state_in = _clone_state(state_template)
        self.return_mtp = return_mtp
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            if return_mtp:
                self.logits, self.state_out, self.mtp = model.step_block(
                    self.token_static, self.state_in, write_memory=True, return_mtp=True
                )
            else:
                self.logits, self.state_out = model.step_block(self.token_static, self.state_in, write_memory=True)

    def __call__(
        self, token_ids: torch.Tensor, state: List[VBlockState]
    ) -> tuple[torch.Tensor, List[VBlockState]] | tuple[torch.Tensor, List[VBlockState], torch.Tensor | None]:
        self.token_static.copy_(token_ids)
        _copy_state(self.state_in, state)
        self.graph.replay()
        if self.return_mtp:
            return self.logits, self.state_out, self.mtp
        return self.logits, self.state_out


def build_graph_step(model, state_template: List[VBlockState], return_mtp: bool = False):
    return build_graph_step_block(model, state_template, token_shape=(1, 1), return_mtp=return_mtp)


def build_graph_step_block(model, state_template: List[VBlockState], token_shape: Tuple[int, int], return_mtp: bool = False):
    try:
        return GraphRunner(model, state_template, token_shape=token_shape, return_mtp=return_mtp)
    except Exception:
        return None
