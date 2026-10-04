from __future__ import annotations

from collections import defaultdict
import csv
import numpy as np
from .types import ContractError


def selected_gain(item, actions):
    f, labels = item.features, item.labels
    nodes = {rid: j for j, rid in enumerate(f.recipe_ids)}
    drops = {slot: i for i, slot in enumerate(f.drop_indices)}
    legal = f.node_mask[:, None] & f.outgoing_mask[None, :]
    g = labels.opportunity(legal)
    if len(actions) != len({(a.recipe_id, a.drop_index) for a in actions}): raise ContractError("Duplicate selection")
    if any(a.configuration_id != f.state_key or a.recipe_id not in nodes or a.drop_index not in drops for a in actions):
        raise ContractError("Selection belongs to another frozen configuration")
    if any(not legal[nodes[a.recipe_id], drops[a.drop_index]] for a in actions): raise ContractError("Selection includes a protected/inactive action")
    return max([float(g[nodes[a.recipe_id], drops[a.drop_indices[0]]]) for a in actions] + [0.])


def object_table(rows):
    aggregate = {}
    for r in rows:
        obj = r["object_id"]
        a = aggregate.setdefault(obj, {"object_id": obj, "family": r["family"], "gain": 0., "reference": 0.,
                                       "states": 0, "no_reference_states": 0, "missing_states": 0})
        a["states"] += 1
        if r.get("missing", False):
            a["missing_states"] += 1; continue
        reference = float(r["reference"])
        if reference <= 0:
            a["no_reference_states"] += 1; continue
        a["gain"] += float(r["gain"]); a["reference"] += reference
    result = []
    for obj in sorted(aggregate):
        a = aggregate[obj]
        a["coverage"] = a["gain"] / a["reference"] if a["reference"] > 0 else None
        a["complete"] = a["missing_states"] == 0
        result.append(a)
    return result


def coverage_summary(rows):
    objects = object_table(rows)
    valid = [o["coverage"] for o in objects if o["coverage"] is not None and o["complete"]]
    near = [o["coverage"] for o in objects if o["coverage"] is not None and o["complete"]
            and o["family"] in {"near", "near_contact", "near-contact"}]
    families = defaultdict(list)
    for o in objects:
        if o["coverage"] is not None and o["complete"]: families[o["family"]].append(o["coverage"])
    return {"objects": objects, "macro": float(np.mean(valid)) if valid else None,
            "near_contact": float(np.mean(near)) if near else None,
            "family_means": {k: float(np.mean(v)) for k, v in families.items()},
            "missing_states": sum(o["missing_states"] for o in objects),
            "zero_reference_states": sum(o["no_reference_states"] for o in objects),
            "independent_objects": len(objects), "positive_complete_objects": len(valid)}


def paired_bootstrap_objects(rows_a, rows_b, repetitions=2000, seed=180018):
    aa = {o["object_id"]: o for o in object_table(rows_a)}
    bb = {o["object_id"]: o for o in object_table(rows_b)}
    if set(aa) != set(bb): raise ValueError("Paired bootstrap needs identical parent objects")
    valid = [obj for obj in sorted(aa) if all(x["coverage"] is not None and x["complete"] for x in (aa[obj], bb[obj]))]
    if not valid:
        return {"status": "NO_POSITIVE_COMPLETE_PAIRS", "n_objects": 0, "interval95": None}
    difference = np.array([aa[o]["coverage"] - bb[o]["coverage"] for o in valid])
    rng = np.random.default_rng(seed)
    samples = difference[rng.integers(0, len(valid), (repetitions, len(valid)))].mean(1)
    return {"status": "TINY_OBJECT_PILOT", "n_objects": len(valid), "repetitions": repetitions,
            "mean_difference": float(difference.mean()), "interval95": np.quantile(samples, [.025, .975]).tolist(),
            "excluded_objects": sorted(set(aa)-set(valid)), "unit": "parent_object"}


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
