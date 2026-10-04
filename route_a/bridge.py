"""Wire adaptation only; all operators, seeds and verification remain shared.

The Integrator's callbacks return arrays and record counts in CostReceipt.
Route A's internal acquisition API carries each array and its receipt together.
No inverse/endpoint/material-injection implementation lives in this module.
"""
from dataclasses import asdict, is_dataclass
import time
import numpy as np
from .audit import assert_physics_allowed
from .recipes import array_hash
from .types import ContractError, value, canonical_hash, readonly

OPS = {"L_rhs": "L", "L_adjoint_rhs": "L_adjoint", "S_rhs": "S",
       "S_adjoint_rhs": "S_adjoint", "B_illumination_rhs": "B"}
ONLINE_HASH_FIELDS = ("chi", "current_polarizability", "points", "r", "forcing", "ell", "Lambda", "source_phase")


class OnlineView:
    """Complete Route A cache identity over the declared readonly online fields."""
    def __init__(self, state):
        self._shared_state = state
        for field in ONLINE_HASH_FIELDS + ("U", "LU", "SU", "D", "step_old"):
            array = np.asarray(value(state, field))
            if array.flags.writeable:
                raise ContractError("Shared FrozenState arrays must be readonly")
        self.state_sha256 = canonical_hash({
            **{k: array_hash(value(state, k)) for k in ONLINE_HASH_FIELDS},
            "geometry": value(state, "receiver_geometry_sha256"),
            "material": value(state, "material_gauge_sha256"),
            "grid_shape": value(state, "grid_shape"),
            "cell_volume": value(state, "cell_volume"), "wavenumber": value(state, "wavenumber"),
            "packing": "four_illumination_Re_then_Im", "current_metric": "shared_whitened"})
        self.U_sha256 = array_hash(value(state, "U"))

    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        return getattr(self._shared_state, name)


def raw_state(state):
    return state._shared_state if isinstance(state, OnlineView) else state


class ReceiptBackend:
    def __init__(self, backend, *, seed_vector_encoding="unit_directions_columns"):
        self.shared = backend; self.seed_vector_encoding = seed_vector_encoding
        self._configuration_hashes = {}
        if seed_vector_encoding not in {"unit_directions_columns", "raw_vectors_columns", "raw_vectors_rows"}:
            raise ContractError("Seed encoding must be explicitly declared")

    def online_view(self, state, audit):
        if isinstance(state, OnlineView): return state
        t = time.perf_counter(); view = OnlineView(state)
        previous = self._configuration_hashes.setdefault(view.configuration_id, view.state_sha256)
        if previous != view.state_sha256:
            raise ContractError("Integrator configuration ID collided across source/material states")
        audit.record({"origin": "route_a.shared_state_identity", "dependencies": list(ONLINE_HASH_FIELDS),
                      "exclusive_seconds": time.perf_counter()-t})
        return view

    def _receipt(self, before, elapsed, origin, dependencies, operation=None):
        delta = self.shared.receipt.delta(before); counters = delta["counters"]
        result = {"origin": origin, "dependencies": list(dependencies), "exclusive_seconds": elapsed,
                  "shared_counters": counters, "shared_wall_seconds": delta["wall_seconds"],
                  "rhs": {op: int(counters[key]) for key, op in OPS.items() if key in counters}}
        if operation is not None: result[operation] = 1
        for k in ("all_bank_materializations", "full_inverse_actions", "full_SVD", "anchor64_builds", "workspace128_builds"):
            if k in counters: result[k] = counters[k]
        if counters.get("geometry_SVD", 0):
            result["full_SVD"] = counters["geometry_SVD"]; result["geometry_preprocessing"] = True
        return result

    def _apply(self, name, vector, operation):
        assert_physics_allowed(); before = self.shared.receipt.snapshot(); t = time.perf_counter()
        output = getattr(self.shared, name)(vector)
        return output, self._receipt(before, time.perf_counter()-t, "shared_backend."+name,
                                     ["declared_current_operator", "current_estimate"], operation)

    def apply_L(self, v): return self._apply("apply_L", v, "L")
    def apply_L_adjoint(self, v): return self._apply("apply_L_adjoint", v, "L_adjoint")
    def apply_S(self, v): return self._apply("apply_S", v, "S")
    def apply_S_adjoint(self, v): return self._apply("apply_S_adjoint", v, "S_adjoint")
    def apply_B(self, v): return self._apply("apply_B", v, "B")

    def get_opm_seeds(self, state):
        assert_physics_allowed(); before = self.shared.receipt.snapshot(); t = time.perf_counter()
        bundle = self.shared.get_opm_seeds(raw_state(state))
        vectors = np.asarray(value(bundle, "vectors"), dtype=np.complex128)
        norms = np.asarray(value(bundle, "norms"), dtype=np.float64)
        if self.seed_vector_encoding != "raw_vectors_rows": vectors = vectors.T
        if self.seed_vector_encoding == "unit_directions_columns":
            active = np.asarray(value(bundle, "active"), dtype=bool)
            if vectors.shape != (6, len(state.U)) or not np.allclose(np.linalg.norm(vectors[active], axis=1), 1., atol=1e-10):
                raise ContractError("Shared SeedBundle unit direction encoding changed")
            vectors = vectors * norms[:, None]
        receipt = self._receipt(before, time.perf_counter()-t, "shared_backend.OPM_seed_bundle",
                                ["current_U", "current_LU", "current_D", "canonical_r", "canonical_forcing", "material_injection"])
        return {"vectors": readonly(vectors), "norms": norms, "source_scales": value(bundle, "source_scales"),
                "active": value(bundle, "active")}, (receipt,)

    def shared_state_features(self, state): return self.shared.shared_state_features(raw_state(state))
    def outgoing_features(self, state): return self.shared.outgoing_features(raw_state(state))

    def verifier(self, plugin, state):
        from shared_backend import VerifiedExchange
        callbacks = {r.id: (lambda rid=r.id: plugin.materialize_selected(rid, state)) for r in plugin.recipes}
        return VerifiedExchange(self.shared, raw_state(state), recipes=callbacks)
