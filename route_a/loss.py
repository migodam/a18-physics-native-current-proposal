from __future__ import annotations

import torch


def expected_best_iid(gains, probabilities, K=4):
    """Exact expectation for iid draws with replacement, not top-K selection."""
    if not isinstance(K, int) or K < 1:
        raise ValueError("Expected-best requires a positive integer draw count")
    if gains.numel() == 0:
        return probabilities.to(torch.float64).sum() * 0
    g, indices = torch.sort(gains.detach().to(torch.float64))
    p = probabilities.to(torch.float64)[indices]
    tails = torch.flip(torch.cumsum(torch.flip(p, [0]), 0), [0]).clamp(0, 1)
    increments = g - torch.cat([torch.zeros(1, dtype=g.dtype, device=g.device), g[:-1]])
    # t * sum_j (1-t)^j equals 1-(1-t)^K and preserves tiny probabilities
    # without a subtractive cancellation or log(0) gradient at t=1.
    chance = tails * sum((1-tails).pow(j) for j in range(K))
    return (increments * chance).sum()


def action_loss(action_logits, noop_logit, gains_fp64, feasible, tau_online, reference_fp64,
                canonical_groups, legal, K=4):
    flat = action_logits.reshape(-1)
    mask = legal.reshape(-1).bool()
    feasible = feasible.reshape(-1).bool() & mask
    raw = gains_fp64.reshape(-1).detach().to(torch.float64)
    tau = tau_online.reshape(-1).detach().to(torch.float64)
    reference = torch.as_tensor(reference_fp64, device=flat.device, dtype=torch.float64).detach()
    if (feasible & (~torch.isfinite(raw) | ~torch.isfinite(tau))).any() or (tau < 0).any():
        raise ValueError("A feasible labelled action needs finite original gain/tolerance")
    if not torch.isfinite(reference) or reference < 0:
        raise ValueError("Invalid detached gain reference")
    if not torch.isfinite(flat[mask]).all() or not torch.isfinite(noop_logit).all():
        raise ValueError("Nonfinite active model logits")
    g = torch.where(feasible & (raw > tau), raw, torch.zeros_like(raw))
    if g.numel() and g.max().item() > reference.item():
        raise ValueError("Common reference excludes an audited action")
    combined = torch.cat([flat.masked_fill(~mask, -torch.inf).to(torch.float64), noop_logit.reshape(1).to(torch.float64)])
    logp = torch.log_softmax(combined, 0)
    p = logp.exp()
    has_positive = bool((g > 0).any())
    if has_positive:
        # Scale before summing: finite large/small gains must not overflow/underflow
        # the gain-mass target. The ratio itself keeps the fixed physical reference.
        scaled = g / g.max()
        target = scaled / scaled.sum()
        positive = target > 0
        list_loss = -(target[positive] * logp[:-1][positive]).sum()
    else:
        list_loss = -logp[-1]
    if reference.item() > 0:
        denom = reference.clamp_min(torch.finfo(torch.float64).tiny)
        # Normalize gains first so subnormal references cannot destroy the
        # expected-best numerator before division. Retain raw FP64 diagnostics.
        cover = 1 - expected_best_iid(torch.cat([g / denom, g.new_zeros(1)]), p, K)
    else:
        cover = combined.new_zeros(())
    duplicate = combined.new_zeros(())
    groups = canonical_groups.reshape(-1)
    if reference.item() > 0:
        for group in torch.unique(groups[mask]):
            members = (groups == group) & mask
            if int(members.sum()) > 1:
                duplicate = duplicate + p[:-1][members].sum().square()
    total = list_loss + cover + .01 * duplicate
    return {"loss": total, "list": list_loss, "cover": cover, "duplicate": duplicate,
            "reference_zero": reference.item() == 0,
            "division_floor_active": 0 < reference.item() < torch.finfo(torch.float64).tiny,
            "numerical_floor_reference": 0 < reference.item() < torch.finfo(torch.float32).tiny}
