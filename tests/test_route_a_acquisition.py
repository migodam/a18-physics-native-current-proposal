"""Synthetic algebra/contract fixtures only. No test below is Maxwell evidence."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
import numpy as np
from route_a.audit import AcquisitionAudit, assert_online_origin, cached_epoch, charged_apply
from route_a.recipes import LazyCache, build_recipes, instantiate, feedback_apply, InvalidCurrent
from route_a.features import extract, Normalizer
from route_a.types import action, ContractError


def fixture():
    rng = np.random.default_rng(180018)
    points = np.array(list(np.ndindex(2, 2, 2)), float); n = 24
    U, _ = np.linalg.qr(rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3)))
    L = np.eye(n) + .025*(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))
    state = SimpleNamespace(U=U, LU=L@U, D=U.conj().T@L@U, chi=np.full(8, .8+.1j),
                            points=points, grid_shape=(2, 2, 2), receiver_mode_ids=(0, 1, 2),
                            protected_slots=(0,), source_phase=np.ones(4, complex),
                            configuration_id="U-initial", object_id="synthetic-object", state_id="synthetic-state",
                            split="train", state_sha256="a"*64, material_gauge_sha256="b"*64,
                            receiver_geometry_sha256="c"*64, cell_volume=1., wavenumber=1.)
    seeds = rng.normal(size=(6, n)) + 1j*rng.normal(size=(6, n))
    seeds -= (seeds@U.conj())@U.T
    class Backend:
        def __init__(self): self.calls = 0; self.seed_calls = 0
        def op(self, v, adj):
            self.calls += 1
            return ((L.conj().T if adj else L)@v,
                    dict(origin="synthetic_unit_test.L", dependencies=["literal_dense_fixture"], exclusive_seconds=0.))
        def apply_L(self, v): return self.op(v, False)
        def apply_L_adjoint(self, v): return self.op(v, True)
        def get_opm_seeds(self, s):
            self.seed_calls += 1
            norms = np.linalg.norm(seeds, axis=1)
            return dict(vectors=seeds, norms=norms, source_scales=2*norms, active=np.ones(6, bool)), [
                dict(origin="synthetic_unit_test.seeds", dependencies=["literal_fixture"], exclusive_seconds=0.)]
        def shared_state_features(self, s):
            return np.zeros(192), [dict(origin="synthetic_unit_test.state", dependencies=["current_estimate"], exclusive_seconds=0.)]
        def outgoing_features(self, s):
            a = np.zeros((3, 32)); a[:, :16] = np.eye(3, 16); a[:, 17] = [1, 2, 3]
            a[:, 31] = 1; a[0, 29] = 1
            return a, [dict(origin="synthetic_unit_test.outgoing", dependencies=["paid_SU_LU"], exclusive_seconds=0.)]
    return state, Backend(), L


class AcquisitionTests(unittest.TestCase):
    def test_registry_and_describe_no_raw_bank(self):
        state, backend, _ = fixture(); cache = LazyCache(); audit = AcquisitionAudit()
        recipes, inactive = build_recipes(state, backend, cache, audit)
        for recipe in recipes: recipe.describe(state)
        self.assertEqual(len(recipes), 98)  # 3 receiver+32 Fourier+21 Haar+24 random+18 feedback
        self.assertEqual(backend.calls, 0); self.assertEqual(audit.calls["raw_currents"], 0)
        self.assertEqual(cache.raw, {}); self.assertEqual(audit.calls["all_bank_materializations"], 0)

    def test_single_recipe_access_and_shared_raw(self):
        s, b, _ = fixture(); cache = LazyCache(); audit = AcquisitionAudit()
        recipes, _ = build_recipes(s, b, cache, audit); recipe = next(r for r in recipes if r.family == "fourier")
        c1 = instantiate(action(recipe.id, 1, s.configuration_id, 0.), s, b, cache, recipes, audit)
        c2 = instantiate(action(recipe.id, 2, s.configuration_id, 0.), s, b, cache, recipes, audit)
        self.assertEqual(len(cache.raw), 1); self.assertEqual(audit.calls["raw_currents"], 1)
        self.assertAlmostEqual(np.linalg.norm(c1), 1.); self.assertAlmostEqual(np.linalg.norm(c2), 1.)
        self.assertLess(np.linalg.norm(np.delete(s.U, 1, 1).conj().T@c1), 1e-12)

    def test_forward_and_adjoint_schur_single_action(self):
        s, b, L = fixture(); P = np.eye(24)-s.U@s.U.conj().T
        A = P@L@P - P@s.LU@np.linalg.solve(s.D, s.U.conj().T@L@P)
        F = P-A; q = np.arange(24) + 1j*np.arange(24)[::-1]
        for adj in (False, True):
            audit = AcquisitionAudit(); before = b.calls
            actual = feedback_apply(q, s, b, audit, adj)
            expected = (F.conj().T if adj else F)@q
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
            self.assertEqual(b.calls-before, 1)

    def test_feedback_prefix_is_lazy(self):
        s, b, _ = fixture(); c = LazyCache(); audit = AcquisitionAudit(); recipes, _ = build_recipes(s, b, c, audit)
        degree2 = next(r for r in recipes if r.id == "A1:feedback:002")
        degree1 = next(r for r in recipes if r.id == "A1:feedback:001")
        degree2.materialize(s, b, c, audit); self.assertEqual(b.calls, 2)
        degree1.materialize(s, b, c, audit); self.assertEqual(b.calls, 2)
        self.assertEqual(len(c.raw), 2)

    def test_protected_and_projected_zero_invalid(self):
        s, b, _ = fixture(); c = LazyCache(); recipes, _ = build_recipes(s, b, c)
        r = next(r for r in recipes if r.id == "A1:receiver:000")
        with self.assertRaises(InvalidCurrent): instantiate(action(r.id, 0, s.configuration_id, 0), s, b, c, recipes)
        with self.assertRaises(InvalidCurrent): instantiate(action(r.id, 1, s.configuration_id, 0), s, b, c, recipes)

    def test_a0_and_stale_action_rejected(self):
        s, b, _ = fixture(); c = LazyCache(); recipes, _ = build_recipes(s, b, c)
        with self.assertRaises(ContractError): action("A0:fourier:000", 1, s.configuration_id, 0)
        with self.assertRaises(ContractError): instantiate(action(recipes[0].id, 1, "old-U", 0), s, b, c, recipes)

    def test_teacher_origin_rejected(self):
        for path in ("cache/teacher_only/x.npz", "D:\\data\\teacher\\x.npz", "workspace128", "anchor64", "full_J"):
            with self.assertRaises(ContractError): assert_online_origin(path)

    def test_cache_invalidated_on_configuration_and_phase(self):
        s, b, _ = fixture(); c = LazyCache(); recipes, _ = build_recipes(s, b, c)
        recipes[4].materialize(s, b, c); first = c.key
        s.configuration_id = "U-after-accept"; c.ensure(s)
        self.assertNotEqual(first, c.key); self.assertEqual(c.raw, {})
        second = c.key; s.source_phase *= 1j; c.ensure(s)
        self.assertNotEqual(second, c.key)

    def test_phase_projection_and_haar_norms(self):
        s, b, _ = fixture(); c = LazyCache(); recipes, _ = build_recipes(s, b, c)
        r = next(r for r in recipes if r.family == "fourier")
        a = action(r.id, 1, s.configuration_id, 0); base = instantiate(a, s, b, c, recipes)
        c.raw[r.id] = r.materialize(s, b, c)*np.exp(.7j)
        rotated = instantiate(a, s, b, c, recipes)
        np.testing.assert_allclose(np.outer(base, base.conj()), np.outer(rotated, rotated.conj()), atol=1e-14)
        for r in recipes:
            if r.family == "haar": self.assertAlmostEqual(np.linalg.norm(r.materialize(s, b, c)), 1.)

    def test_features_and_train_only_normalizer(self):
        s, b, _ = fixture(); f, recipes, audit = extract(s, b)
        self.assertEqual(f.xi.shape, (204,)); self.assertEqual(f.nodes.shape[1], 32); self.assertEqual(f.outgoing.shape, (3, 32))
        self.assertFalse(f.outgoing_mask[0]); self.assertEqual(b.calls, 0)
        self.assertFalse(any(f.node_mask[j] for j,r in enumerate(recipes) if r.family == "receiver"))
        self.assertFalse(f.xi.flags.writeable); self.assertEqual(audit.calls["raw_currents"], 0)
        normalizer = Normalizer.fit([f]); out = normalizer.transform(f)
        self.assertTrue(np.isfinite(out.xi).all()); self.assertEqual(normalizer.training_object_ids, (s.object_id,))
        s.split = "validation"; vf, _, _ = extract(s, b)
        with self.assertRaises(ContractError): Normalizer.fit([vf])

    def test_cached_epoch_blocks_physics(self):
        s, b, _ = fixture()
        with self.assertRaises(ContractError):
            with cached_epoch(): charged_apply(b, "apply_L", np.ones(24), AcquisitionAudit())
        self.assertEqual(b.calls, 0)


if __name__ == "__main__": unittest.main()
