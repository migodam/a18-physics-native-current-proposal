from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
from .cache_data import load_dataset, file_hash, dataset_fingerprint
from .features import Normalizer
from .metrics import coverage_summary, paired_bootstrap_objects, selected_gain, write_csv
from .model import ActionMLP
from .train import cached_policy_rows, authorize
from .types import ContractError, SCHEMA_HASH


def evaluate_cached(model, normalizer, items, policy, K=4):
    from .baselines import analytic_actions
    rows = cached_policy_rows(model, items, normalizer, K)
    control = []
    for item in items:
        actions = analytic_actions(item.features, **policy, K=K)
        control.append({"object_id": item.features.object_id, "family": item.family,
                        "gain": selected_gain(item, actions), "reference": item.labels.reference_fp64})
    return {"scope": "cached_label_policy_evaluation", "coverage": coverage_summary(rows),
            "paired_bootstrap": paired_bootstrap_objects(rows, control),
            "online_cost": "NOT_MEASURED", "deployment_gate": "NOT_RUN"}, rows


def main():
    parser = argparse.ArgumentParser(description="Frozen, Integrator-authorized policy evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "locked_test"], required=True)
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--features", default="cache/online_features"); parser.add_argument("--labels", default="labels/action_A1.jsonl")
    parser.add_argument("--frozen-manifest", required=True); parser.add_argument("--evaluation-authorisation")
    parser.add_argument("--output", default="results/route_a/frozen_evaluation")
    args = parser.parse_args(); frozen = json.loads(Path(args.frozen_manifest).read_text())
    if frozen.get("checkpoint_sha256") != file_hash(args.checkpoint) or frozen.get("manifest_sha256") != file_hash(args.manifest):
        raise ContractError("Frozen checkpoint/data identity mismatch")
    if args.split == "locked_test":
        if not args.evaluation_authorisation: raise ContractError("Integrator must open the unified test window")
        auth = json.loads(Path(args.evaluation_authorisation).read_text())
        if auth.get("owner") != "integrator" or auth.get("frozen_manifest_sha256") != file_hash(args.frozen_manifest):
            raise ContractError("Wrong frozen test authorization")
        claim = Path(auth["route_a_once_claim"])
        claim.mkdir(parents=True, exist_ok=False)  # A second evaluation fails closed.
        (claim / "owner.json").write_text(json.dumps({"checkpoint": frozen["checkpoint_sha256"], "window_id": auth["window_id"]}) + "\n")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_hash") != SCHEMA_HASH: raise ContractError("Checkpoint feature schema mismatch")
    normalizer = Normalizer.from_dict(payload["normalizer"])
    if normalizer.fingerprint != payload["normalizer_hash"]: raise ContractError("Frozen normalization drift")
    model = ActionMLP(); model.load_state_dict(payload["weights"])
    # The shared new manifest uses train/validation/test. locked_test names the
    # authorization protocol, not a fourth or retrospectively reassigned split.
    items = load_dataset(args.manifest, args.features, args.labels,
                         "test" if args.split == "locked_test" else args.split)
    if args.split == "locked_test" and any(i.manifest.get("historical_exposure", True) for i in items):
        raise ContractError("Previously exposed objects cannot be advertised as blind test")
    report, rows = evaluate_cached(model, normalizer, items, payload["analytic_baseline"]["policy"])
    report.update(checkpoint_sha256=frozen["checkpoint_sha256"], split=args.split, dataset_hash=dataset_fingerprint(items))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "cached_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(out / "state_coverage.csv", rows)
    write_csv(out / "object_coverage.csv", report["coverage"]["objects"])
    print(json.dumps({"scope": report["scope"], "deployment_gate": "NOT_RUN", "output": str(out)}))


if __name__ == "__main__": main()
