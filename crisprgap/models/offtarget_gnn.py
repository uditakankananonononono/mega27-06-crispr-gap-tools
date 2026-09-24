"""GNN scorer for guide/off-target duplexes.

Each duplex is a graph: positions are nodes (one-hot base pair identity + mismatch
flag + position), edges connect adjacent positions and all mismatch positions to
each other (mismatch interaction graph). A small message-passing GNN produces a
cleavage logit. This attacks the mismatch-context gap (gap 5).
"""
from __future__ import annotations

import torch
from torch import nn

from crisprgap.sequence import BASE_TO_IDX

# node feature dim: 16 (guide-target base pair one-hot, 4x4) + 1 mismatch + 1 position frac
NODE_DIM = 18


def duplex_to_graph(guide: str, offtarget: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (node_features (L, NODE_DIM), edge_index (2, E)) for a duplex."""
    g, o = guide.upper(), offtarget.upper()
    assert len(g) == len(o)
    L = len(g)
    x = torch.zeros(L, NODE_DIM)
    for i, (a, b) in enumerate(zip(g, o)):
        ai, bi = BASE_TO_IDX.get(a), BASE_TO_IDX.get(b)
        if ai is not None and bi is not None:
            x[i, ai * 4 + bi] = 1.0
        x[i, 16] = 0.0 if (ai is not None and ai == bi) else 1.0
        x[i, 17] = i / max(L - 1, 1)
    edges: list[tuple[int, int]] = []
    for i in range(L - 1):
        edges += [(i, i + 1), (i + 1, i)]
    mm = [i for i in range(L) if x[i, 16] > 0.5]
    for i in mm:
        for j in mm:
            if i != j:
                edges.append((i, j))
    if not edges:  # single-node graph fallback: self-loop
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


class MessagePassingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.msg = nn.Linear(dim * 2, dim)
        self.upd = nn.GRUCell(dim, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = torch.relu(self.msg(torch.cat([x[src], x[dst]], dim=1)))
        agg = torch.zeros_like(x).index_add_(0, dst, m)
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=torch.float)).clamp(min=1).unsqueeze(1)
        return self.upd(agg / deg, x)


class OffTargetGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, hidden: int = 48, steps: int = 3):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden) for _ in range(steps)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0)
        return self.head(pooled).squeeze(-1)
