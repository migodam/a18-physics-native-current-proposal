from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping
import json
import numpy as np

SCHEMA = "a18-route-a-feature-v1"
STATE_NORM_INDICES = (32, *range(36, 48), 51, 52, 59, 60, 61, *range(192, 198))
FEATURE_SCHEMA = {"version": SCHEMA, "dimensions": [204, 32, 32],
                  "state_norm_log1p_scale1": STATE_NORM_INDICES,
                  "node_raw_norm": 21, "outgoing_raw_norms": [16, 17],
                  "atom_seed_projections": False,
                  "families": ["receiver", "fourier", "haar", "random", "feedback"]}
SCHEMA_HASH = sha256(json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ContractError(ValueError):
    pass


class BackendUnavailable(RuntimeError):
    pass


def value(obj: Any, name: str, default: Any = ...):
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is ...:
        raise ContractError(f"Missing declared online field: {name}")
    return default


def readonly(array, dtype=None):
    a = np.array(array, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class ActionSpec:
    """A-only fallback with the Integrator's declared wire fields.

    It contains no physics and can be converted to shared_backend.ActionSpec.
    """
    route: str
    action_id: str
    configuration_id: str
    drop_indices: tuple[int, ...]
    prior_logit: float
    recipe_id: str

    @property
    def drop_index(self):
        if len(self.drop_indices) != 1:
            raise ContractError("Route A v1 only supports singleton exchange")
        return self.drop_indices[0]

    def __post_init__(self):
        if self.route != "A" or not self.recipe_id.startswith("A1:"):
            raise ContractError("A0 labels/IDs cannot enter the A1 action interface")
        if len(self.drop_indices) != 1 or self.drop_indices[0] < 0:
            raise ContractError("Invalid singleton drop")
        if not self.configuration_id or not np.isfinite(self.prior_logit):
            raise ContractError("Missing configuration or nonfinite prior")

    def to_shared(self):
        try:
            from shared_backend import ActionSpec as SharedAction
        except ImportError as exc:
            raise BackendUnavailable("shared_backend.ActionSpec not installed") from exc
        return SharedAction(**self.__dict__)


def action(recipe_id, drop_index, configuration_id, logit):
    key = f"{configuration_id}|{recipe_id}|{int(drop_index)}"
    return ActionSpec("A", sha256(key.encode()).hexdigest(), configuration_id,
                      (int(drop_index),), float(logit), recipe_id)


@dataclass(frozen=True)
class FeatureBatch:
    xi: np.ndarray
    nodes: np.ndarray
    outgoing: np.ndarray
    node_mask: np.ndarray
    outgoing_mask: np.ndarray
    recipe_ids: tuple[str, ...]
    drop_indices: tuple[int, ...]
    state_key: str
    object_id: str = ""
    state_id: str = ""
    split: str = ""
    receipts: tuple[dict, ...] = ()
    schema_hash: str = SCHEMA_HASH
    provenance: tuple[dict, ...] = ()

    def __post_init__(self):
        for key in ("xi", "nodes", "outgoing"):
            object.__setattr__(self, key, readonly(getattr(self, key), np.float64))
        for key in ("node_mask", "outgoing_mask"):
            object.__setattr__(self, key, readonly(getattr(self, key), bool))
        n, k = len(self.recipe_ids), len(self.drop_indices)
        shapes = {"xi": (204,), "nodes": (n, 32), "outgoing": (k, 32),
                  "node_mask": (n,), "outgoing_mask": (k,)}
        for key, shape in shapes.items():
            a = getattr(self, key)
            if a.shape != shape or not np.isfinite(a).all():
                raise ContractError(f"Invalid {key}: expected {shape}, finite values")
        if n > 122 or k > 16 or len(set(self.recipe_ids)) != n:
            raise ContractError("Recipe/slot cap or duplicate IDs")
        if len(set(self.drop_indices)) != k or any(i < 0 for i in self.drop_indices):
            raise ContractError("Invalid outgoing IDs")
        if any(not s.startswith("A1:") for s in self.recipe_ids):
            raise ContractError("A0 and A1 identities cannot be mixed")
        if not self.state_key or self.schema_hash != SCHEMA_HASH:
            raise ContractError("Configuration/schema identity is required")


@dataclass(frozen=True)
class LabelBatch:
    gains_fp64: np.ndarray
    feasible: np.ndarray
    tau_online: np.ndarray
    reference_fp64: float
    canonical_groups: np.ndarray
    complete: bool = True

    def __post_init__(self):
        for key in ("gains_fp64", "tau_online"):
            object.__setattr__(self, key, readonly(getattr(self, key), np.float64))
        object.__setattr__(self, "feasible", readonly(self.feasible, bool))
        object.__setattr__(self, "canonical_groups", readonly(self.canonical_groups, np.int64))
        shape = self.gains_fp64.shape
        if len(shape) != 2 or any(getattr(self, k).shape != shape for k in
                ("feasible", "tau_online", "canonical_groups")):
            raise ContractError("Action-level labels need matched recipe/drop matrices")
        # A completed invalid endpoint record has undefined raw utility; retain
        # NaN in the offline matrix and map it to opportunity zero only through
        # its explicit infeasible flag. Missing records still fail in the loader.
        if (self.feasible & (~np.isfinite(self.gains_fp64) | ~np.isfinite(self.tau_online))).any():
            raise ContractError("A feasible action needs finite original gain/tolerance")
        if (self.tau_online < 0).any() or not np.isfinite(self.reference_fp64) or self.reference_fp64 < 0:
            raise ContractError("Invalid original tolerance/reference")
        if not self.complete:
            raise ContractError("Incomplete action labels cannot train a drop-conditioned head")

    def opportunity(self, legal=None):
        valid = self.feasible & (self.gains_fp64 > self.tau_online)
        if legal is not None:
            valid &= legal
        return np.where(valid, self.gains_fp64, 0.0)


@dataclass(frozen=True)
class DatasetItem:
    features: FeatureBatch
    labels: LabelBatch
    family: str
    manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels.gains_fp64.shape != self.features.nodes.shape[:1] + self.features.outgoing.shape[:1]:
            raise ContractError("Feature/label shape mismatch")
        legal = self.features.node_mask[:, None] & self.features.outgoing_mask[None, :]
        best = self.labels.opportunity(legal).max(initial=0.)
        if self.labels.reference_fp64 < best:
            raise ContractError("Common detached reference must include all audited A1 actions")


def canonical_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
