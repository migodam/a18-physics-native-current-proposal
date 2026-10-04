from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
from .baselines import analytic_actions, all_registered_policies
from .cache_data import load_dataset, file_hash
from .metrics import coverage_summary, selected_gain, write_csv
from .types import ContractError

MACHINERY = ("all_bank_materializations", "full_inverse_actions", "anchor64_builds", "workspace128_builds")


def assess(items, timing_rows, g0_receipt=None, config_sha256=None):
    """Use actual teacher/reference labels and independently measured wall rows."""
    if len(items) > 12 or len({i.features.object_id for i in items}) > 6:
        raise ContractError("Development preflight cap is 6 objects/12 states")
    if any(i.features.split != "development" for i in items):
        raise ContractError("Preflight cannot use held-out objects")
    if not items: raise ContractError("Empty development evidence cannot pass a gate")
    g0 = "NOT_RUN"
    if g0_receipt:
        status = g0_receipt.get("gates", {}).get("G0", "NOT_RUN")
        if isinstance(status, dict): status = status.get("status")
        if (status == "PASS" and g0_receipt.get("evidence_scope") == "real_vector_maxwell"
                and config_sha256 and g0_receipt.get("config_sha256") == config_sha256):
            g0 = "PASS"
        else: g0 = "HOLD"
    oracle = []
    for item in items:
        legal = item.features.node_mask[:, None] & item.features.outgoing_mask[None, :]
        best = item.labels.opportunity(legal).max(initial=0.)
        oracle.append({"object_id": item.features.object_id, "state_id": item.features.state_id,
                       "configuration_id": item.features.state_key, "family": item.family,
                       "gain": float(best), "reference": item.labels.reference_fp64,
                       "active_actions": int(legal.sum()), "invalid_actions": int((legal & ~item.labels.feasible).sum()),
                       "undefined_raw_gain_actions": int((legal & ~np.isfinite(item.labels.gains_fp64)).sum()),
                       "positive_actions": int((item.labels.opportunity(legal) > 0).sum()),
                       "original_dictionary_reference": item.manifest.get("original_dictionary_gain_fp64")})
    oracle_summary = coverage_summary(oracle)
    original_rows = []
    for row in oracle:
        ref = row["original_dictionary_reference"]
        if ref is not None and (not np.isfinite(ref) or ref < 0): raise ContractError("Invalid original dictionary reference")
        if ref is not None and ref > row["reference"]: raise ContractError("Common union reference excludes original dictionary")
        original_rows.append(dict(row, reference=ref if ref is not None else 0., missing=ref is None))
    original_summary = coverage_summary(original_rows)
    policies = []
    for policy in all_registered_policies():
        rows = [{"object_id": i.features.object_id, "family": i.family,
                 "gain": selected_gain(i, analytic_actions(i.features, **policy)),
                 "reference": i.labels.reference_fp64} for i in items]
        policies.append({"policy": policy, "metrics": coverage_summary(rows)})
    eligible = [p for p in policies if p["metrics"]["macro"] is not None]
    strongest = max(eligible, key=lambda p: p["metrics"]["macro"]) if eligible else None
    timings = {}
    for r in timing_rows:
        key = (r["state_id"], r["configuration_id"])
        if key in timings: raise ContractError("Duplicate matched timing evidence")
        timings[key] = r
    ratios = []; timing_missing = []
    for item in items:
        key = (item.features.state_id, item.features.state_key); row = timings.get(key)
        if row is None or row.get("scope") != "real_online_matched_timing" or not row.get("complete"):
            timing_missing.append(key); continue
        if not row.get("source_sha256") or not row.get("config_sha256"):
            raise ContractError("Timing needs source/config identity")
        if config_sha256 and row["config_sha256"] != config_sha256:
            raise ContractError("Timing belongs to another config")
        if not all(np.isfinite(float(row[k])) for k in ("T_acq_Dfull", "T_acq_A1")):
            raise ContractError("Nonfinite matched times")
        if float(row["T_acq_Dfull"]) <= 0 or float(row["T_acq_A1"]) < 0:
            raise ContractError("Invalid matched acquisition times")
        counters = row.get("A1_counters", {})
        if any(k not in counters for k in MACHINERY) or "full_SVD" not in counters:
            timing_missing.append(key); continue
        if any(counters[k] != 0 for k in MACHINERY):
            raise ContractError("Online A1 used forbidden machinery")
        if counters["full_SVD"] and row.get("geometry_SVD_separately_charged") is not True:
            raise ContractError("Online SVD is only allowed as separately charged geometry preprocessing")
        ratios.append(float(row["T_acq_A1"]) / float(row["T_acq_Dfull"]))
    macro, near = oracle_summary["macro"], oracle_summary["near_contact"]
    coverage = macro is not None and near is not None and macro >= .85 and near >= .70
    cost = len(ratios) == len(items) and bool(ratios) and np.median(ratios) <= .5 and np.quantile(ratios, .95) <= 1
    original_missing = any(r["original_dictionary_reference"] is None for r in oracle)
    real = all(i.manifest.get("evidence_scope") == "real_vector_maxwell" for i in items)
    g1 = "HOLD" if (g0 != "PASS" or not real or timing_missing or original_missing
                     or macro is None or near is None) else ("PASS" if coverage and cost else "FAIL")
    headroom = strongest is not None and strongest["metrics"]["macro"] < .95
    g2 = "NOT_RUN" if g1 != "PASS" else ("PASS" if headroom else "ANALYTIC_ONLY")
    result = {"scope": "development_exposed_selection_not_unbiased_generalization", "gates": {"G0": g0, "G1A": g1, "G2": g2},
              "evidence_scope": "real_vector_maxwell" if real and g0 == "PASS" else "UNVERIFIED_SHARED_INPUTS",
              "config_sha256": config_sha256,
              "oracle": oracle_summary, "oracle_against_original_dictionary": original_summary,
              "strongest_analytic": strongest, "all_analytic_policies": policies,
              "timing_missing": [list(k) for k in timing_missing], "original_reference_missing": original_missing,
              "acquisition_ratio_median": float(np.median(ratios)) if ratios else None,
              "acquisition_ratio_p95": float(np.quantile(ratios, .95)) if ratios else None,
              "training_authorized": g1 == "PASS" and g2 == "PASS"}
    return result, oracle


def main():
    parser = argparse.ArgumentParser(description="A1 action-level development oracle/cost preflight")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--config", default="config/pilot.json")
    parser.add_argument("--g0-receipt", default="results/backend/G0.json")
    parser.add_argument("--features", default="cache/online_features"); parser.add_argument("--labels", default="labels/action_A1.jsonl")
    parser.add_argument("--timings", default="results/route_a/preflight_online_timings.jsonl")
    parser.add_argument("--output", default="results/route_a/preflight")
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    required = [Path(x) for x in (args.config, args.g0_receipt, args.manifest, args.features, args.labels, args.timings)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        result = {"status": "HOLD_SHARED_INPUTS_UNAVAILABLE", "gates": {"G0": "NOT_RUN", "G1A": "NOT_RUN", "G2": "NOT_RUN"},
                  "missing": missing, "training_authorized": False, "maxwell_evidence": False}
        (out / "GATE_DECISION.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result)); return 2
    items = load_dataset(args.manifest, args.features, args.labels, "development")
    timings = [json.loads(line) for line in Path(args.timings).read_text().splitlines() if line.strip()]
    result, rows = assess(items, timings, json.loads(Path(args.g0_receipt).read_text()), file_hash(args.config))
    (out / "GATE_DECISION.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(out / "oracle_state_coverage.csv", rows)
    write_csv(out / "oracle_object_coverage.csv", result["oracle"]["objects"])
    print(json.dumps({"gates": result["gates"], "output": str(out)}))
    return 0 if result["training_authorized"] else 2


if __name__ == "__main__": raise SystemExit(main())
