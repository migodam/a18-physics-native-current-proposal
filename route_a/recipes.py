from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import time
import numpy as np
from .audit import AcquisitionAudit, charged_apply, assert_physics_allowed, acquisition_timing
from .types import ContractError, canonical_hash, readonly, value

FAMILIES = ("receiver", "fourier", "haar", "random", "feedback")
CAPS = (16, 32, 32, 24, 18)
SEEDS = ("O1", "O2", "P1", "P2", "M1", "M2")
RECIPE_SCHEMA_HASH = canonical_hash({"version": "a18-lazy-recipes-v1", "caps": CAPS,
    "families": FAMILIES, "seed_order": SEEDS, "degrees": [0, 1, 2], "random_seed": 180018,
    "random_generator": "NumPy.Philox.SeedSequence(seed,index)",
    "fourier_order": "integer_norm2_lexicographic_then_xyz", "haar": "Cartesian_C_unpadded_dyadic",
    "projection": "retained_core_two_MGS_relative_floor_1e-10"})


class InvalidCurrent(ContractError):
    pass


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = sha256(str((a.shape, a.dtype.str)).encode()); h.update(a.view(np.uint8))
    return h.hexdigest()


def cache_key(state):
    u_hash = array_hash(value(state, "U"))
    if value(state, "U_sha256", u_hash) != u_hash:
        raise ContractError("Declared U hash does not match current U")
    return canonical_hash({"state_id": value(state, "state_id"),
        "state_sha256": value(state, "state_sha256"),
        "configuration_id": value(state, "configuration_id"),
        "U_hash": u_hash,
        "material_gauge": value(state, "material_gauge_sha256"),
        "B_hash": value(state, "B_gauge_sha256", value(state, "material_gauge_sha256")),
        "geometry": value(state, "receiver_geometry_sha256"),
        "phase": {"real": np.asarray(value(state, "source_phase")).real.tolist(),
                  "imag": np.asarray(value(state, "source_phase")).imag.tolist()},
        "recipe_schema": RECIPE_SCHEMA_HASH,
        "seed_schema": "OPM-six-degree-012-v1",
        "seed_floor": 1e-10, "projection_floor": 1e-10,
        "orthogonality_floor": 1e-10, "arnoldi_breakdown_floor": 1e-12,
        "current_metric": "whitened"})


class LazyCache:
    def __init__(self):
        self.key = None
        self.raw = {}
        self.prefixes = {}
        self.seed_bundle = None
        self.seed_receipts = ()

    def ensure(self, state, audit=None):
        started = time.perf_counter()
        key = cache_key(state)
        invalidated = self.key is not None and key != self.key
        if key != self.key:
            self.key = key
            self.raw.clear(); self.prefixes.clear()
            self.seed_bundle = None; self.seed_receipts = ()
        if audit is not None:
            audit.record({"origin": "route_a.cache_validation", "dependencies": ["frozen_state_identity", "current_U", "material_gauge", "source_phase"],
                          "exclusive_seconds": time.perf_counter()-started, "cache_invalidated": invalidated})
        return key


def seeds_for(state, backend, cache, audit):
    assert_physics_allowed()
    cache.ensure(state, audit)
    if cache.seed_bundle is None:
        result = backend.get_opm_seeds(state)
        if isinstance(result, tuple) and len(result) == 2:
            bundle, receipts = result
        else:
            bundle = result
            receipts = value(bundle, "receipts")
        vectors = value(bundle, "vectors")
        if isinstance(vectors, dict):
            vectors = np.stack([vectors[name] for name in SEEDS])
        vectors = np.asarray(vectors, dtype=np.complex128)
        norms = np.asarray(value(bundle, "norms"), dtype=np.float64)
        scales = np.asarray(value(bundle, "source_scales"), dtype=np.float64)
        active = np.asarray(value(bundle, "active"), dtype=bool)
        n = np.asarray(value(state, "U")).shape[0]
        if vectors.shape != (6, n) or norms.shape != (6,) or scales.shape != (6,) or active.shape != (6,):
            raise ContractError("SeedBundle must use ordered O1/O2/P1/P2/M1/M2")
        if not all(np.isfinite(a).all() for a in (vectors, norms, scales)) or (scales < 0).any():
            raise ContractError("Invalid seed scales")
        actual = np.linalg.norm(vectors, axis=1)
        if not np.allclose(actual[active], norms[active], rtol=1e-10, atol=0):
            raise ContractError("Seed norm receipt does not match its current")
        expected = (scales > 0) & (norms > 1e-10 * scales)
        if not np.array_equal(active, expected):
            raise ContractError("Original relative seed floor must be respected")
        for r in receipts:
            audit.record(r)
        cache.seed_receipts = tuple(receipts)
        cache.seed_bundle = {"vectors": readonly(vectors), "norms": readonly(norms),
                             "source_scales": readonly(scales), "active": readonly(active)}
    return cache.seed_bundle


def feedback_apply(q, state, backend, audit, adjoint=False):
    """One fresh L or L* action, using already-paid LU and D.

    Aeff=P L (I-U D^-1 U* L) P.  Its adjoint can be applied as
    P L* (q-U D^-* (LU)*q), avoiding a separate L*U construction.
    """
    U = np.asarray(value(state, "U")); LU = np.asarray(value(state, "LU"))
    D = np.asarray(value(state, "D"))
    q = np.asarray(q, dtype=np.complex128)
    q = q - U @ (U.conj().T @ q)
    try:
        if adjoint:
            h = np.linalg.solve(D.conj().T, LU.conj().T @ q)
            l, _ = charged_apply(backend, "apply_L_adjoint", q - U @ h, audit)
        else:
            l, _ = charged_apply(backend, "apply_L", q, audit)
            h = np.linalg.solve(D, U.conj().T @ l)
            l = l - LU @ h
    except np.linalg.LinAlgError as exc:
        raise InvalidCurrent("Shared retained compression is singular; no ridge added") from exc
    a = l - U @ (U.conj().T @ l)
    return q - a


@dataclass(frozen=True)
class Recipe:
    id: str
    family: str
    static_descriptor: dict
    dependency_ids: tuple[str, ...]

    def describe(self, state):
        """Descriptors contain metadata only; never construct an atom here."""
        return dict(self.static_descriptor)

    def materialize(self, state, backend, lazy_cache, audit=None):
        assert_physics_allowed()
        audit = audit or AcquisitionAudit()
        with acquisition_timing(audit, f"route_a.recipes/{self.id}", self.dependency_ids,
                                "selected_raw_materialization"):
            lazy_cache.ensure(state, audit)
            return self._materialize_impl(state, backend, lazy_cache, audit)

    def _materialize_impl(self, state, backend, lazy_cache, audit):
        if self.id in lazy_cache.raw:
            return lazy_cache.raw[self.id]
        if not self.id.startswith("A1:") or self.family not in FAMILIES:
            raise ContractError("Only A1 lazy recipes can be materialized")
        d = self.static_descriptor; points = np.asarray(value(state, "points"))
        n = len(points); out = np.zeros((n, 3), dtype=np.complex128)
        if self.family == "receiver":
            mode = int(d["geometry_index"])
            modes = value(state, "receiver_geometry_modes", None)
            if modes is not None:
                raw = np.asarray(modes)[:, mode].copy()
            else:
                ids = tuple(value(state, "receiver_mode_ids"))
                if mode not in ids:
                    raise InvalidCurrent("Receiver geometry mode no longer cached")
                raw = np.asarray(value(state, "U"))[:, ids.index(mode)].copy()
        elif self.family == "fourier":
            box_min = np.asarray(d["box_min"]); lengths = np.asarray(d["box_lengths"])
            x = (points - box_min) / np.where(lengths > 0, lengths, 1)
            out[:, d["component"]] = np.exp(2j * np.pi * (x @ np.asarray(d["frequency"]))) / np.sqrt(n)
            raw = out.ravel()
        elif self.family == "haar":
            shape = tuple(value(state, "grid_shape")); starts = d["starts"]; ends = d["ends"]
            factors = []
            for start, end, detail in zip(starts, ends, d["detail"]):
                size = end - start
                if detail:
                    left = size // 2; right = size - left
                    if left == 0:
                        raise InvalidCurrent("Unsupported Haar detail on a singleton axis")
                    v = np.r_[np.full(left, np.sqrt(right / (left * size))),
                              np.full(right, -np.sqrt(left / (right * size)))]
                else:
                    v = np.full(size, 1 / np.sqrt(size))
                factors.append(v)
            atom = np.einsum("i,j,k->ijk", *factors)
            grid = out.reshape(shape + (3,))
            grid[tuple(slice(a, b) for a, b in zip(starts, ends)) + (d["component"],)] = atom
            raw = out.ravel()
        elif self.family == "random":
            rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([180018, d["random_index"]])))
            raw = np.array([1, -1, 1j, -1j])[rng.integers(0, 4, size=3 * n)] / np.sqrt(3 * n)
        else:
            bundle = seeds_for(state, backend, lazy_cache, audit)
            sid, degree = d["seed_id"], d["degree"]
            if not bundle["active"][sid]:
                raise InvalidCurrent("Inactive relative-floor O/P/M seed")
            prefix = (sid, 0)
            if prefix not in lazy_cache.prefixes:
                lazy_cache.prefixes[prefix] = bundle["vectors"][sid]
            for depth in range(1, degree + 1):
                prefix = (sid, depth)
                if prefix not in lazy_cache.prefixes:
                    q = feedback_apply(lazy_cache.prefixes[(sid, depth - 1)], state, backend, audit, sid < 2)
                    if np.linalg.norm(q) <= 1e-10 * bundle["source_scales"][sid]:
                        raise InvalidCurrent("Feedback recipe is below relative norm floor")
                    lazy_cache.prefixes[prefix] = readonly(q)
            raw = np.asarray(lazy_cache.prefixes[(sid, degree)]).copy()
        if raw.shape != (3 * n,) or not np.isfinite(raw).all():
            raise InvalidCurrent("Invalid raw current shape/value")
        lazy_cache.raw[self.id] = readonly(raw)
        audit.count("raw_currents")
        return lazy_cache.raw[self.id]


def _base(state):
    points = np.asarray(value(state, "points"), dtype=float)
    chi = np.asarray(value(state, "chi"))
    if points.ndim != 2 or points.shape[1] != 3 or chi.size != len(points):
        raise ContractError("Declared current grid and chi estimate do not match")
    low = points.min(axis=0); lengths = points.max(axis=0) - low
    return points, chi.ravel(), low, lengths


def build_recipes(state, backend, cache, audit=None):
    audit = audit or AcquisitionAudit(); started = time.perf_counter()
    before = len(audit.receipts); points, chi, low, lengths = _base(state)
    common = {"center": ((low + lengths / 2)).tolist(), "scale": lengths.tolist(),
              "box_min": low.tolist(), "box_lengths": lengths.tolist(),
              "support_fraction": 1., "degree": 0, "seed_id": -1, "raw_norm": 1.}
    recipes = []; inactive = {}
    modes = value(state, "receiver_geometry_modes", None)
    if modes is not None:
        indices = range(min(16, np.asarray(modes).shape[1]))
    else:
        indices = sorted({int(i) for i in value(state, "receiver_mode_ids", ()) if int(i) >= 0})[:16]
    for i in indices:
        desc = dict(common, geometry_index=int(i))
        recipes.append(Recipe(f"A1:receiver:{i:03d}", "receiver", desc, ("paid_receiver_geometry_modes",)))
    if not list(indices):
        inactive["receiver"] = "no already-paid geometry modes"
    frequencies = []
    for radius in range(5):
        shell = sorted((x for x in product(range(-radius, radius + 1), repeat=3)
                        if max(map(abs, x)) == radius), key=lambda x: (sum(i*i for i in x), x))
        frequencies.extend(shell)
    frequencies = sorted(set(frequencies), key=lambda x: (sum(i*i for i in x), x))
    fourier_count = 0
    for freq in frequencies:
        for component in range(3):
            if fourier_count == 32: break
            desc = dict(common, frequency=freq, component=component)
            recipes.append(Recipe(f"A1:fourier:{fourier_count:03d}", "fourier", desc, ("physical_grid",)))
            fourier_count += 1
        if fourier_count == 32: break
    shape = tuple(int(i) for i in value(state, "grid_shape"))
    regular = len(shape) == 3 and np.prod(shape) == len(points)
    if regular:
        grid_points = points.reshape(shape + (3,))
        # Require Cartesian C ordering, without padding or changing physical support.
        for a in range(3):
            selector = [0, 0, 0]; selector[a] = slice(None)
            axis_values = grid_points[tuple(selector) + (a,)]
            target_shape = [1, 1, 1]; target_shape[a] = shape[a]
            regular &= bool(np.allclose(grid_points[..., a], axis_values.reshape(target_shape)))
    count = 0
    if regular:
        for level in range(8):
            tiles = 2**level
            if tiles > max(shape): break
            for tile in product(range(tiles), repeat=3):
                starts = tuple(size * i // tiles for size, i in zip(shape, tile))
                ends = tuple(size * (i + 1) // tiles for size, i in zip(shape, tile))
                sizes = np.subtract(ends, starts)
                if (sizes <= 0).any(): continue
                for detail in product((0, 1), repeat=3):
                    if not any(detail) or any(bit and size < 2 for bit, size in zip(detail, sizes)): continue
                    sl = tuple(slice(a, b) for a, b in zip(starts, ends))
                    local = grid_points[sl].reshape(-1, 3)
                    for component in range(3):
                        desc = dict(common, level=level, tile=tile, starts=starts, ends=ends, detail=detail,
                                    component=component, center=local.mean(axis=0).tolist(),
                                    scale=(local.max(axis=0)-local.min(axis=0)).tolist(),
                                    support_fraction=float(np.prod(sizes)/len(points)))
                        recipes.append(Recipe(f"A1:haar:{count:03d}", "haar", desc, ("physical_grid",)))
                        count += 1
                        if count == 32: break
                    if count == 32: break
                if count == 32: break
            if count == 32: break
    if count == 0: inactive["haar"] = "grid does not support Cartesian Haar details"
    for i in range(24):
        recipes.append(Recipe(f"A1:random:{i:03d}", "random", dict(common, random_index=i), ("fixed_Philox_seed_180018",)))
    bundle = seeds_for(state, backend, cache, audit)
    for sid, seed in enumerate(SEEDS):
        for degree in range(3):
            dep = ("current_U", f"seed_{seed}") + (("paid_LU", "L_adjoint" if sid < 2 else "L") if degree else ())
            desc = dict(common, degree=degree, seed_id=sid, active=bool(bundle["active"][sid]),
                        raw_norm=float(bundle["norms"][sid]) if degree == 0 else 0., raw_norm_valid=degree == 0)
            recipes.append(Recipe(f"A1:feedback:{sid*3+degree:03d}", "feedback", desc, dep))
    if len(recipes) > 122: raise ContractError("Recipe cap exceeded")
    nested = sum(float(r["exclusive_seconds"]) for r in audit.receipts[before:])
    audit.record({"origin": "route_a.recipe_descriptors", "dependencies": ["grid", "current_chi_estimate", "paid_geometry", "OPM_seed_bundle"],
                  "exclusive_seconds": max(0., time.perf_counter()-started-nested), "inactive_families": inactive})
    return tuple(recipes), inactive


def instantiate(action, state, backend, cache, recipes, audit=None):
    audit = audit or AcquisitionAudit()
    with acquisition_timing(audit, "route_a.retained_projection_normalization",
                            ["current_retained_U", action.recipe_id], "selected_action_instantiation"):
        return _instantiate(action, state, backend, cache, recipes, audit)


def _instantiate(action, state, backend, cache, recipes, audit):
    if action.configuration_id != value(state, "configuration_id"):
        raise ContractError("Action belongs to another current configuration")
    i = action.drop_indices[0]
    U = np.asarray(value(state, "U"))
    protected = set(value(state, "protected_slots", ()))
    if i in protected or not 0 <= i < U.shape[1]:
        raise InvalidCurrent("Protected/nonexistent outgoing slot")
    lookup = {r.id: r for r in recipes}
    if action.recipe_id not in lookup:
        raise ContractError("Recipe schema/ID mismatch")
    raw = lookup[action.recipe_id].materialize(state, backend, cache, audit)
    V = np.delete(U, i, axis=1); c = raw.copy()
    for _ in range(2):
        c -= V @ (V.conj().T @ c)
    norm = np.linalg.norm(c)
    if not np.isfinite(norm) or norm <= 1e-10 * np.linalg.norm(raw):
        raise InvalidCurrent("Projected recipe has zero/relative-floor norm")
    return readonly(c / norm)
