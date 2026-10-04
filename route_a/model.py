from __future__ import annotations

import numpy as np
import torch
from torch import nn
from .types import action


class ActionMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        if hidden != 64:
            raise ValueError("The preregistered first-round model uses hidden=64")
        self.node = nn.Sequential(nn.Linear(236, 64), nn.LayerNorm(64), nn.GELU(),
                                  nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(224, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 1))
        self.noop = nn.Linear(128, 1)

    def forward(self, xi, nodes, outgoing, node_mask, outgoing_mask):
        batch, n, _ = nodes.shape; k = outgoing.shape[1]
        h = self.node(torch.cat([nodes, xi[:, None, :].expand(-1, n, -1)], dim=-1))
        weights = node_mask[..., None].to(h.dtype)
        mean = (h * weights).sum(1) / weights.sum(1).clamp_min(1)
        if n:
            maximum = h.masked_fill(~node_mask[..., None], -torch.inf).max(1).values
            maximum = torch.where(node_mask.any(1)[:, None], maximum, torch.zeros_like(maximum))
        else:
            maximum = torch.zeros_like(mean)
        pool = torch.cat([mean, maximum], -1)
        hh = h[:, :, None, :].expand(-1, -1, k, -1)
        pp = pool[:, None, None, :].expand(-1, n, k, -1)
        oo = outgoing[:, None, :, :].expand(-1, n, -1, -1)
        logits = self.head(torch.cat([hh, pp, oo], -1)).squeeze(-1)
        legal = node_mask[:, :, None] & outgoing_mask[:, None, :]
        return logits.masked_fill(~legal, -torch.inf), self.noop(pool).squeeze(-1)

    @torch.no_grad()
    def propose(self, features, K=4):
        device = next(self.parameters()).device
        arrays = [features.xi, features.nodes, features.outgoing, features.node_mask, features.outgoing_mask]
        tensors = [torch.as_tensor(np.array(a, copy=True), device=device,
                                   dtype=torch.bool if i >= 3 else torch.float32).unsqueeze(0)
                   for i, a in enumerate(arrays)]
        logits, noop = self(*tensors)
        return select_actions(features, logits[0].cpu().numpy(), float(noop[0]), K)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def select_actions(features, logits, noop_logit, K=4):
    if not 0 <= K <= 4:
        raise ValueError("The frozen pilot permits at most four atomic proposals")
    logits = np.asarray(logits)
    expected = (len(features.recipe_ids), len(features.drop_indices))
    if logits.shape != expected or not np.isfinite(noop_logit):
        raise ValueError("Logit shape/no-op invalid")
    ranked = []
    for j, rid in enumerate(features.recipe_ids):
        for i, drop in enumerate(features.drop_indices):
            score = logits[j, i]
            if features.node_mask[j] and features.outgoing_mask[i]:
                if not np.isfinite(score):
                    raise ValueError("An active action has nonfinite logit")
                ranked.append((float(score), rid, int(drop)))
    ranked.sort(key=lambda a: (-a[0], a[1], a[2]))
    if not ranked or noop_logit >= ranked[0][0]:
        return []
    return [action(rid, drop, features.state_key, score) for score, rid, drop in ranked[:K]]
