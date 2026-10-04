from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from .audit import AcquisitionAudit, assert_online_origin
from .recipes import FAMILIES, LazyCache, build_recipes, seeds_for
from .types import ContractError, FeatureBatch, canonical_hash, value, readonly, STATE_NORM_INDICES

STATE_SOURCES = (["current_chi_estimate"]*13 + ["current_polarizability"]*2 + ["grid_volume"] +
                 ["physical_box"]*6 + ["receiver_geometry"]*5 + ["illumination_ids"] +
                 ["wavenumber", "physical_grid", "active_cells", "paid_geometry_cache"] +
                 ["source_phase_canonicalized_r"]*8 + ["forcing"]*4 + ["retained_current_residual"]*4 +
                 ["current_U", "material_dimension", "Lambda", "ell", "step_old", "D", "normal_residual",
                  "D", "D", "D", "D", "SU", "LU", "cached_reduced_gradient", "accepted_exchanges", "cached_full_output"] +
                 ["canonical_receiver_illumination_r_sketch"]*64 + ["current_chi_Fourier_sketch"]*64 +
                 ["current_U_OPM_seed_norms"]*6 + ["relative_seed_floor"]*6)


def feature_provenance(block, values, sources, receipt_indices, audit, node_index=None):
    """Every channel references a measured group receipt, charged once.

    Shared extractors may refine timing to channel groups via feature_indices.
    Without that detail we report the resolution as the whole extractor call;
    we never invent per-channel microseconds or sum the same cost repeatedly.
    """
    rows = []
    for column, source in enumerate(sources):
        refs = [j for j in receipt_indices if "feature_indices" not in audit.receipts[j]
                or column in audit.receipts[j]["feature_indices"]]
        if not refs: raise ContractError("A feature has no measured dependency receipt")
        rows.append({"block": block, "row": node_index, "column": column,
                     "origin": source, "dependencies": sorted({d for j in refs for d in audit.receipts[j]["dependencies"]}),
                     "cost_receipt_indices": refs,
                     "exclusive_seconds": sum(float(audit.receipts[j]["exclusive_seconds"]) for j in refs),
                     "timing_resolution": "shared_group_reference_counted_once"})
    return rows


def _feature_result(result, shape, audit):
    if not isinstance(result, tuple) or len(result) != 2:
        raise ContractError("Shared feature extractor must return (features, dependency_receipts)")
    array, receipts = result; array = np.asarray(array, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"Shared feature shape must be {shape}")
    for receipt in receipts:
        audit.record(receipt)
    return array


def node_descriptor(recipe, state):
    d = recipe.describe(state); x = np.zeros(32, dtype=np.float64)
    x[FAMILIES.index(recipe.family)] = 1
    x[5:8] = d["center"]; x[8:11] = d["scale"]
    if "component" in d: x[11 + d["component"]] = 1
    if "frequency" in d: x[14:17] = d["frequency"]
    x[17] = d["degree"]
    sid = d["seed_id"]
    if sid >= 0: x[18 + sid // 2] = 1
    x[21] = d["raw_norm"]; x[22] = d["support_fraction"]
    chi = np.asarray(value(state, "chi")).ravel()
    if "starts" in d:
        shape = tuple(value(state, "grid_shape"))
        chi = chi.reshape(shape)[tuple(slice(a, b) for a, b in zip(d["starts"], d["ends"]))].ravel()
    if chi.size:
        x[23:26] = [chi.real.mean(), chi.imag.mean(), np.abs(chi).std()]
        x[30] = 1
    x[26] = sid; x[27] = d.get("geometry_index", 0)
    x[28] = 1; x[29] = "frequency" in d; x[31] = recipe.family == "receiver"
    return x


def extract(state, backend, cache=None, audit=None):
    """Only Integrator's online view is accepted; teachers are separate objects."""
    cache = cache or LazyCache(); audit = audit or AcquisitionAudit()
    started = time.perf_counter(); before = len(audit.receipts)
    assert_online_origin(value(state, "online_provenance", {"origin": "FrozenState.current_estimate"}))
    U = np.asarray(value(state, "U")); k = U.shape[1]
    if k > 16 or not np.allclose(U.conj().T @ U, np.eye(k), rtol=0, atol=1e-10):
        raise ContractError("Feature contract requires a whitened orthonormal rank<=16 current basis")
    mark = len(audit.receipts)
    xi192 = _feature_result(backend.shared_state_features(state), (192,), audit)
    state_refs = list(range(mark, len(audit.receipts)))
    mark = len(audit.receipts)
    outgoing = _feature_result(backend.outgoing_features(state), (k, 32), audit)
    outgoing_refs = list(range(mark, len(audit.receipts)))
    mark = len(audit.receipts)
    recipes, inactive = build_recipes(state, backend, cache, audit)
    bundle = seeds_for(state, backend, cache, audit)
    seed_refs = list(range(mark, len(audit.receipts)))
    # Raw cached seed logs use scale=1. Train-only normalization stores/replays
    # the six nonzero median scales, never fits to validation/test objects.
    xi = np.r_[xi192, np.log1p(bundle["norms"]), bundle["active"].astype(float)]
    node_rows = []; node_refs = []
    for recipe in recipes:
        node_started = time.perf_counter()
        node_rows.append(node_descriptor(recipe, state))
        node_refs.append(len(audit.receipts))
        audit.record({"origin": f"route_a.node_descriptor/{recipe.id}",
                      "dependencies": [*recipe.dependency_ids, "recipe_metadata", "current_chi_estimate"],
                      "exclusive_seconds": time.perf_counter()-node_started})
    nodes = np.stack(node_rows) if recipes else np.empty((0, 32))
    mask = np.array([r.static_descriptor.get("active", True) for r in recipes])
    protected = set(value(state, "protected_slots", ()))
    outgoing_mask = np.array([i not in protected and bool(outgoing[i, 31]) for i in range(k)])
    nested = sum(float(r["exclusive_seconds"]) for r in audit.receipts[before:])
    audit.record({"origin": "route_a.feature_assembly", "dependencies": ["shared_state_features", "outgoing_features", "current_chi_estimate", "recipe_metadata"],
                  "exclusive_seconds": max(0., time.perf_counter()-started-nested),
                  "atom_seed_projections": False, "inactive_families": inactive})
    assembly = len(audit.receipts)-1
    packing_started = time.perf_counter()
    provenance = feature_provenance("xi", xi, STATE_SOURCES[:192], state_refs, audit)
    for p in feature_provenance("xi", xi, STATE_SOURCES[192:], seed_refs+[assembly], audit):
        p["column"] += 192; provenance.append(p)
    node_sources = (["recipe_family"]*5 + ["recipe_support_center"]*3 + ["recipe_support_scale"]*3 +
                    ["recipe_Cartesian_orientation"]*3 + ["recipe_Fourier_wavevector"]*3 + ["recipe_degree"] +
                    ["OPM_family"]*3 + ["static_or_cached_raw_norm", "recipe_support_fraction"] +
                    ["current_chi_estimate_on_declared_support"]*3 + ["seed_ID", "paid_receiver_mode_ID"] + ["recipe_valid_flag"]*4)
    outgoing_sources = (["slot_ID"]*16 + ["paid_LU", "paid_SU", "paid_D", "paid_D"] + ["paid_D"]*4 +
                        ["paid_SU_and_canonical_r"]*2 + ["existing_material_injection_cache"]*2 +
                        ["receiver_mode_ID", "protected_slot", "injection_valid", "slot_valid"])
    for j in range(len(recipes)):
        provenance.extend(feature_provenance("nodes", nodes[j], node_sources, [node_refs[j]], audit, j))
    for j in range(k):
        provenance.extend(feature_provenance("outgoing", outgoing[j], outgoing_sources, outgoing_refs, audit, j))
    features = FeatureBatch(xi, nodes, outgoing, mask, outgoing_mask,
                            tuple(r.id for r in recipes), tuple(range(k)),
                            value(state, "configuration_id"), value(state, "object_id"),
                            value(state, "state_id"), value(state, "split", "development"),
                            tuple(audit.receipts), provenance=tuple(provenance))
    audit.record({"origin": "route_a.provenance_and_output_packing",
                  "dependencies": ["feature_sources", "cost_receipts", "feature_arrays"],
                  "exclusive_seconds": time.perf_counter()-packing_started})
    # Finish construction before exposing the frozen object to callers.
    object.__setattr__(features, "receipts", tuple(audit.receipts))
    return features, recipes, audit


@dataclass(frozen=True)
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    node_mean: np.ndarray
    node_std: np.ndarray
    outgoing_mean: np.ndarray
    outgoing_std: np.ndarray
    state_norm_scales: np.ndarray
    node_norm_scale: float
    outgoing_norm_scales: np.ndarray
    training_object_ids: tuple[str, ...]

    def __post_init__(self):
        for key, shape in (("state_mean", (204,)), ("state_std", (204,)), ("node_mean", (32,)),
                           ("node_std", (32,)), ("outgoing_mean", (32,)), ("outgoing_std", (32,)),
                           ("state_norm_scales", (len(STATE_NORM_INDICES),)), ("outgoing_norm_scales", (2,))):
            array = readonly(getattr(self, key), np.float64)
            if array.shape != shape or not np.isfinite(array).all(): raise ContractError("Malformed normalization")
            if ("std" in key or "scales" in key) and (array <= 0).any(): raise ContractError("Invalid normalization scale")
            object.__setattr__(self, key, array)
        if not np.isfinite(self.node_norm_scale) or self.node_norm_scale <= 0: raise ContractError("Invalid node norm scale")

    @classmethod
    def fit(cls, features):
        features = list(features)
        if not features or any(f.split != "train" or not f.object_id for f in features):
            raise ContractError("Normalization only fits named training objects")
        norms = np.stack([np.expm1(f.xi[list(STATE_NORM_INDICES)]) for f in features])
        if (norms < 0).any(): raise ContractError("Declared state norm logs cannot be negative")
        scales = _nonzero_medians(norms)
        states = np.stack([_state_norm_scale(f.xi, scales) for f in features])
        nodes = np.concatenate([f.nodes[f.node_mask] for f in features])
        outgoing = np.concatenate([f.outgoing[f.outgoing_mask] for f in features])
        node_scale = _nonzero_medians(nodes[:, 21:22])[0]
        outgoing_scales = _nonzero_medians(outgoing[:, 16:18])
        nodes = nodes.copy(); outgoing = outgoing.copy()
        nodes[:, 21] = np.log1p(nodes[:, 21] / node_scale)
        outgoing[:, 16:18] = np.log1p(outgoing[:, 16:18] / outgoing_scales)
        sm, ss = _stats(states, 204, [31, 63, *range(198, 204)])
        nm, ns = _stats(nodes, 32, [*range(5), *range(11, 14), *range(18, 21), *range(28, 32)])
        om, os = _stats(outgoing, 32, [*range(16), 29, 30, 31])
        return cls(sm, ss, nm, ns, om, os, scales, float(node_scale), outgoing_scales,
                   tuple(sorted({f.object_id for f in features})))

    def transform(self, f):
        xi = (_state_norm_scale(f.xi, self.state_norm_scales) - self.state_mean) / self.state_std
        nodes = f.nodes.copy(); outgoing = f.outgoing.copy()
        if (nodes[:, 21] < 0).any() or (outgoing[:, 16:18] < 0).any(): raise ContractError("Norm features cannot be negative")
        nodes[:, 21] = np.log1p(nodes[:, 21] / self.node_norm_scale)
        outgoing[:, 16:18] = np.log1p(outgoing[:, 16:18] / self.outgoing_norm_scales)
        nodes = (nodes - self.node_mean) / self.node_std
        outgoing = (outgoing - self.outgoing_mean) / self.outgoing_std
        nodes[~f.node_mask] = 0; outgoing[~f.outgoing_mask] = 0
        return FeatureBatch(xi, nodes, outgoing, f.node_mask, f.outgoing_mask,
                            f.recipe_ids, f.drop_indices, f.state_key, f.object_id, f.state_id,
                            f.split, f.receipts, provenance=f.provenance)

    def as_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else list(v) if isinstance(v, tuple) else v
                for k, v in self.__dict__.items()}

    @property
    def fingerprint(self):
        return canonical_hash(self.as_dict())

    @classmethod
    def from_dict(cls, d):
        arrays = {k: np.asarray(v) if isinstance(v, list) else v for k, v in d.items() if k != "training_object_ids"}
        return cls(**arrays, training_object_ids=tuple(d["training_object_ids"]))


def _state_norm_scale(xi, scales):
    a = np.array(xi, copy=True)
    norms = np.expm1(a[list(STATE_NORM_INDICES)])
    if (norms < 0).any(): raise ContractError("Declared state norm logs cannot be negative")
    a[list(STATE_NORM_INDICES)] = np.log1p(norms / scales)
    return a


def _nonzero_medians(norms):
    if not np.isfinite(norms).all() or (norms < 0).any(): raise ContractError("Invalid cached norm")
    return np.array([np.median(v[v > 0]) if np.any(v > 0) else 1. for v in norms.T])


def _stats(data, dimension, discrete):
    if data.size:
        mean = data.mean(axis=0); std = data.std(axis=0)
        std[std == 0] = 1
    else:
        mean = np.zeros(dimension); std = np.ones(dimension)
    mean[discrete] = 0; std[discrete] = 1
    return mean, std
