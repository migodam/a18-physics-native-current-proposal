from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import numpy as np
import torch
from .cache_data import file_hash
from .features import Normalizer, extract
from .model import ActionMLP
from .recipes import LazyCache, instantiate
from .audit import AcquisitionAudit
from .types import BackendUnavailable, ContractError, SCHEMA_HASH, value


class CheckpointPolicy:
    def __init__(self, path, expected_sha256, device="cpu"):
        if file_hash(path) != expected_sha256: raise ContractError("Checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema_hash"] != SCHEMA_HASH: raise ContractError("Feature schema changed")
        self.normalizer = Normalizer.from_dict(payload["normalizer"])
        if self.normalizer.fingerprint != payload["normalizer_hash"]:
            raise ContractError("Normalizer changed")
        self.model = ActionMLP().to(device); self.model.load_state_dict(payload["weights"]); self.model.eval()

    def propose(self, features, K=4):
        return self.model.propose(self.normalizer.transform(features), K)


class RouteAPlugin:
    """Registers selected materialization with the shared verifier.

    Endpoint solves, original thresholds, full Jd/cache updates and new states
    are provided by Integrator. This plugin contains no Maxwell solver.
    """
    route = "A"

    def __init__(self, backend):
        self.backend = backend
        self.cache = LazyCache(); self.state = None; self.recipes = (); self.audit = None

    def prepare(self, state):
        audit = AcquisitionAudit()
        if hasattr(self.backend, "online_view"): state = self.backend.online_view(state, audit)
        features, self.recipes, audit = extract(state, self.backend, self.cache, audit)
        self.state = state; self.audit = audit
        return features, audit

    def materialize_selected(self, recipe_id, state=None):
        """Raw selected callback; shared verifier owns projection/normalization."""
        state = self.state if state is None else state
        if hasattr(self.backend, "online_view"): state = self.backend.online_view(state, self.audit)
        if state.configuration_id != self.state.configuration_id: raise ContractError("Prepare the current configuration first")
        recipe = next((r for r in self.recipes if r.id == recipe_id), None)
        if recipe is None: raise ContractError("Unknown selected recipe")
        return recipe.materialize(state, self.backend, self.cache, self.audit)

    def instantiate(self, action, state=None, audit=None):
        state = state if state is not None else self.state
        if state is None: raise ContractError("Prepare the current immutable state first")
        if self.state is None or value(state, "configuration_id") != value(self.state, "configuration_id"):
            raise ContractError("Prepare the new current configuration before materialization")
        if self.cache.key is None: self.prepare(state)
        return instantiate(action, state, self.backend, self.cache, self.recipes, audit or self.audit)


def import_factory(spec):
    try:
        module, name = spec.split(":", 1)
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise BackendUnavailable(f"Integrator entrypoint is unavailable: {spec}") from exc


def gain_record(record):
    row = asdict(record) if is_dataclass(record) else dict(record)
    if "feasible" not in row:
        raise ContractError("Missing endpoint feasibility")
    if not row["feasible"]:
        return row
    required = {"feasible", "gain_fp64", "tau_online", "current_stability_relative", "normal_residual_relative"}
    if not required.issubset(row):
        raise ContractError("Shared verifier must expose original FP64 gain and numerical gates")
    if not all(np.isfinite(float(row[k])) for k in required - {"feasible"}):
        raise ContractError("Nonfinite verifier result")
    if row["tau_online"] < 0:
        raise ContractError("Invalid original tolerance")
    return row


def normalize_shared_result(result):
    if isinstance(result, dict) and "status" in result:
        if "cost" not in result: raise ContractError("Every failed/successful verification needs a receipt")
        if result["status"] != "OK":
            return {"feasible": False, "invalid_reason": result.get("reason", result["status"])}
        row = {"feasible": True, "gain_fp64": result["gain"], "tau_online": result["tau"],
               "current_stability_relative": result["stability"]["relative"],
               "normal_residual_relative": result["normal_relative_residual"]}
        if bool(result["accepted"]) != bool(row["gain_fp64"] > row["tau_online"]):
            raise ContractError("Shared accept flag disagrees with its original finite utility/tolerance")
        return gain_record(row)
    return gain_record(value(result, "gain", result))


def verify_policy(plugin, state, actions, verified_exchange):
    """One initial-state proposal round; Integrator separately owns acceptance.

    The verifier calls plugin.instantiate via its registered route registry.
    Every attempted/failed action stays in the ledger; no replacement candidate.
    """
    if (len(actions) > 4 or len({a.action_id for a in actions}) != len(actions)
            or len({(a.recipe_id, tuple(a.drop_indices)) for a in actions}) != len(actions)):
        raise ContractError("Too many/duplicate atomic finalists")
    attempts = []; best = None
    for a in actions:
        if a.configuration_id != value(state, "configuration_id"):
            raise ContractError("Stale action configuration")
        result = verified_exchange.evaluate(a)
        # Integrator returns its GainRecord plus endpoint/cost evidence. Adapters
        # normalize that envelope; no local finite-gain reimplementation exists.
        g = normalize_shared_result(result)
        # Feasibility already belongs to the original shared numerical gates.
        # Do not introduce a second local threshold/endpoint policy.
        accepted = bool(g["feasible"] and g["gain_fp64"] > g["tau_online"])
        attempts.append({"action": asdict(a) if is_dataclass(a) else dict(a),
                         "gain": g, "positive_original_verifier": accepted,
                         "cost": value(result, "cost", None)})
        if accepted and (best is None or g["gain_fp64"] > best["gain"]["gain_fp64"]):
            best = attempts[-1]
    return {"attempts": attempts, "best": best, "no_op": best is None}
