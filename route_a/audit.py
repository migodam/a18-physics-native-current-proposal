from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import time
import contextvars
from contextlib import contextmanager
import numpy as np
from .types import ContractError

FORBIDDEN = ("teacher_only", "/teacher/", "teacher_labels", "full_j", "full_h", "full_optimal_step",
             "ground_truth", "anchor64", "workspace128", "exact_future_gain")
_EPOCH = contextvars.ContextVar("route_a_cached_epoch", default=None)


def assert_physics_allowed():
    counter = _EPOCH.get()
    if counter is not None:
        counter["attempted_physics_actions"] += 1
        raise ContractError("Physics acquisition is forbidden inside cached training epochs")


@contextmanager
def cached_epoch():
    counter = {"physics_actions": 0, "attempted_physics_actions": 0}
    token = _EPOCH.set(counter)
    try:
        yield counter
        if counter["attempted_physics_actions"]:
            raise ContractError("An epoch attempted physics access")
    finally:
        _EPOCH.reset(token)


@contextmanager
def acquisition_timing(audit, origin, dependencies, scope):
    """Record exclusive local work once, including failed and cached attempts."""
    started = time.perf_counter(); before = len(audit.receipts); status = "complete"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        nested = sum(float(r["exclusive_seconds"]) for r in audit.receipts[before:])
        audit.record({"origin": origin, "dependencies": list(dependencies),
                      "exclusive_seconds": max(0., time.perf_counter()-started-nested),
                      "scope": scope, "status": status})


def assert_online_origin(obj):
    """Validate provenance, not the identity of legitimate numerical arrays."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for key, item in obj.items():
            assert_online_origin(str(key))
            assert_online_origin(item)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            assert_online_origin(item)
    elif isinstance(obj, (str, Path)):
        text = str(obj).lower().replace("\\", "/")
        if any(name in text for name in FORBIDDEN):
            raise ContractError(f"Forbidden online provenance: {text}")


def receipt_dict(receipt):
    r = asdict(receipt) if is_dataclass(receipt) else dict(receipt)
    if "exclusive_seconds" not in r or "origin" not in r or "dependencies" not in r:
        raise ContractError("Every acquisition needs origin/dependencies/exclusive_seconds")
    t = float(r["exclusive_seconds"])
    if not np.isfinite(t) or t < 0:
        raise ContractError("Invalid exclusive cost receipt")
    return r


class AcquisitionAudit:
    def __init__(self, scope="online_A1"):
        self.scope = scope
        self.calls = dict.fromkeys(("all_bank_materializations", "raw_currents", "L", "L_adjoint",
                                   "S", "S_adjoint", "B", "full_inverse_actions", "full_SVD",
                                   "anchor64_builds", "workspace128_builds"), 0)
        self.rhs = {}
        self.receipts = []

    def record(self, receipt, *, online=True):
        r = receipt_dict(receipt)
        if online:
            assert_online_origin({k: r[k] for k in ("origin", "dependencies")})
            if r.get("full_inverse_actions", 0) or r.get("all_bank_materializations", 0):
                raise ContractError("Full inverse/bank is forbidden in online acquisition")
            if r.get("full_SVD", 0) and not r.get("geometry_preprocessing", False):
                raise ContractError("Only separately charged receiver geometry SVD is allowed")
        if any(int(r.get(key, 0)) < 0 for key in self.calls):
            raise ContractError("Negative machinery counters are invalid")
        self.receipts.append(r)
        for key in self.calls:
            self.calls[key] += int(r.get(key, 0))
        for operation, rhs in r.get("rhs", {}).items():
            if int(rhs) < 0: raise ContractError("Negative RHS count")
            self.rhs[operation] = self.rhs.get(operation, 0) + int(rhs)

    def count(self, operation, rhs=1):
        self.calls[operation] = self.calls.get(operation, 0) + 1
        self.rhs[operation] = self.rhs.get(operation, 0) + int(rhs)

    def snapshot(self):
        return {"scope": self.scope, "calls": self.calls.copy(), "rhs": self.rhs.copy(),
                "exclusive_seconds": sum(float(r["exclusive_seconds"]) for r in self.receipts),
                "receipts": self.receipts.copy()}


def charged_apply(backend, name, vector, audit):
    assert_physics_allowed()
    t = time.perf_counter()
    result = getattr(backend, name)(vector)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ContractError(f"{name} must return (array, CostReceipt)")
    output, receipt = result
    operation = {"apply_L": "L", "apply_L_adjoint": "L_adjoint",
                 "apply_S": "S", "apply_S_adjoint": "S_adjoint", "apply_B": "B"}[name]
    n_rhs = 1 if np.asarray(vector).ndim == 1 else np.asarray(vector).shape[-1]
    audit.record(receipt)
    r = receipt_dict(receipt)
    if not r.get(operation, 0):
        audit.calls[operation] += 1
    if operation not in r.get("rhs", {}):
        audit.rhs[operation] = audit.rhs.get(operation, 0) + n_rhs
    if not np.isfinite(np.asarray(output)).all():
        raise ContractError(f"Nonfinite {name} result")
    return np.asarray(output), time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser(description="Static dependency audit; actual receipts come from Integrator")
    parser.add_argument("--config", default="config/pilot.json")
    parser.add_argument("--split", default="development", choices=["development"])
    parser.add_argument("--output", default="results/route_a/dependency_audit.json")
    parser.add_argument("--state-npz", help="Integrator-supplied whitelisted physical state for CPU interface smoke")
    parser.add_argument("--object-id"); parser.add_argument("--state-id"); parser.add_argument("--receiver-count", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    a = config["route_A"]
    if (a["state_features"], a["node_features"], a["outgoing_features"]) != (204, 32, 32):
        raise ContractError("Feature contract changed")
    if args.state_npz:
        if not args.object_id or not args.state_id: parser.error("Real-state smoke requires explicit object/state IDs")
        from shared_backend.fft_maxwell import load_backend
        from .bridge import ReceiptBackend
        from .runtime import RouteAPlugin
        from .types import action
        from .recipes import RECIPE_SCHEMA_HASH
        assert_online_origin(args.state_npz)
        backend = load_backend(args.state_npz, device="cpu")
        U = backend.get_receiver_basis(args.receiver_count)
        state = backend.freeze_state(U, object_id=args.object_id, state_id=args.state_id, split="development",
                                     receiver_mode_ids=tuple(range(args.receiver_count)))
        plugin = RouteAPlugin(ReceiptBackend(backend)); features, audit = plugin.prepare(state)
        legal = [i for i in features.drop_indices if features.outgoing_mask[i]]
        rid = next((r for r in features.recipe_ids if r.startswith("A1:feedback:") and
                    features.node_mask[features.recipe_ids.index(r)] and int(r.split(":")[-1]) % 3 == 1), None)
        selected = None
        if legal and rid:
            before = set(plugin.cache.raw); current = plugin.instantiate(action(rid, legal[0], state.configuration_id, 0.))
            selected = {"recipe_id": rid, "drop_index": legal[0], "norm": float(np.linalg.norm(current)),
                        "new_raw_recipe_ids": sorted(set(plugin.cache.raw)-before)}
            if len(plugin.cache.raw) != 1: raise ContractError("Single selection acquired other recipes")
        result = {"scope": "real_state_CPU_interface_smoke_only", "status": "PASS_INTERFACE_ONLY",
                  "object_id": state.object_id, "state_id": state.state_id, "historical_exposure": True,
                  "source_sha256": backend.metadata["public_source_sha256"], "configuration_id": state.configuration_id,
                  "dimensions": [len(features.xi), features.nodes.shape[1], features.outgoing.shape[1]],
                  "family_counts": {family: sum(r.family == family for r in plugin.recipes) for family in a["recipes"]},
                  "recipe_schema_hash": RECIPE_SCHEMA_HASH, "selected": selected,
                  "feature_provenance_entries": len(features.provenance), "acquisition": audit.snapshot(),
                  "shared_total_cost": backend.receipt.snapshot(), "G0": "SEPARATE_INTEGRATOR_EVIDENCE",
                  "G1A": "NOT_RUN", "coverage": None, "deployment_timing": "NOT_MEASURED", "training_authorized": False}
        p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"status": result["status"], "family_counts": result["family_counts"], "output": str(p)}))
        return
    result = {"scope": "static_dependency_contract_only", "status": "IMPLEMENTED_NOT_PHYSICS_TESTED",
              "G0": "NOT_RUN", "G1A": "NOT_RUN", "family_caps": a["recipes"],
              "max_recipes": 122, "atom_seed_projections": False,
              "forbidden_online": list(FORBIDDEN), "backend_required": "shared_backend",
              "dynamic_machinery_counters": "NOT_MEASURED", "split": args.split}
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(p)}))


if __name__ == "__main__":
    main()
