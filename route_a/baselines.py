from __future__ import annotations

import numpy as np
from .types import action

ORDERS = (("receiver", "fourier", "haar", "random", "feedback"),
          ("feedback", "receiver", "fourier", "haar", "random"))


def analytic_actions(features, *, order=0, degree=0, drop_rule=0, K=4, kind="analytic"):
    """Twelve fixed policies: two family orders, three degrees, two drop rules.

    Each family cursor follows frozen ID order; at most K paired atomic actions,
    never an incoming-by-drop cross product or a gain-based online ranking.
    """
    if not 0 <= K <= 4 or order not in (0, 1) or degree not in (0, 1, 2) or drop_rule not in (0, 1):
        raise ValueError("Unregistered analytic policy")
    if kind not in {"analytic", "cheap_fixed", "opm_balanced"}:
        raise ValueError("Unregistered control")
    legal = [i for i in range(len(features.drop_indices)) if features.outgoing_mask[i]]
    if drop_rule == 0:
        legal.sort(key=lambda i: (features.outgoing[i, 17], features.drop_indices[i]))
    else:
        legal.sort(key=lambda i: -features.drop_indices[i])
    if not legal or K == 0:
        return []
    if kind == "opm_balanced":
        # Fixed O1/P1/M1/O2 ordering; no gain or operator look-ahead.
        chosen = []
        for sid in (0, 2, 4, 1):
            candidates = sorted(rid for j, rid in enumerate(features.recipe_ids)
                                if features.node_mask[j] and rid.startswith("A1:feedback:")
                                and int(features.nodes[j, 26]) == sid
                                and int(features.nodes[j, 17]) == 0)
            if candidates: chosen.append(candidates[0])
        return [action(rid, features.drop_indices[legal[i % len(legal)]], features.state_key, 0.)
                for i, rid in enumerate(chosen[:K])]
    families = {family: [] for family in ORDERS[order]}
    for j, rid in enumerate(features.recipe_ids):
        family = rid.split(":")[1]
        if not features.node_mask[j]: continue
        if kind == "cheap_fixed" and family == "feedback": continue
        if family == "feedback" and int(features.nodes[j, 17]) != degree: continue
        families[family].append(rid)
    for ids in families.values(): ids.sort()
    chosen = []; cursors = {f: 0 for f in families}
    while len(chosen) < K:
        added = False
        for family in ORDERS[order]:
            ids = families[family]; i = cursors[family]
            if i < len(ids):
                chosen.append(ids[i]); cursors[family] += 1; added = True
                if len(chosen) == K: break
        if not added: break
    return [action(rid, features.drop_indices[legal[i % len(legal)]], features.state_key, 0.) for i, rid in enumerate(chosen)]


def all_registered_policies():
    analytic = [dict(order=o, degree=d, drop_rule=r) for o in range(2) for d in range(3) for r in range(2)]
    return analytic + [dict(kind="cheap_fixed"), dict(kind="opm_balanced")]
