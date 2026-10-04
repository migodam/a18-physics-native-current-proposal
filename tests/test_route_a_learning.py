"""Synthetic CPU tests, never a neural Maxwell pilot or scientific gate PASS."""
import itertools
from dataclasses import replace
import tempfile
from pathlib import Path
import json
import unittest
import numpy as np
import torch
from route_a.model import ActionMLP, select_actions
from route_a.loss import action_loss, expected_best_iid
from route_a.types import FeatureBatch, LabelBatch, DatasetItem, ContractError
from route_a.metrics import coverage_summary, paired_bootstrap_objects
from route_a.baselines import analytic_actions
from route_a.preflight import assess
from route_a.train import authorize, pack_batch

torch.set_num_threads(2)


def features(n=3, k=2, split="train", obj="synthetic-a"):
    xi = np.zeros(204); nodes = np.zeros((n, 32)); nodes[:, 1] = 1
    outgoing = np.zeros((k, 32)); outgoing[:, 31] = 1
    return FeatureBatch(xi, nodes, outgoing, np.ones(n, bool), np.ones(k, bool),
                        tuple(f"A1:fourier:{i:03d}" for i in range(n)), tuple(range(k)),
                        "synthetic-U", obj, obj+":state", split,
                        (dict(origin="synthetic_unit_test", dependencies=[], exclusive_seconds=0.),))


class LearningTests(unittest.TestCase):
    def test_exact_iid_expected_max_and_ties(self):
        for gains in ([0., .2, .7], [0., .7, .7]):
            p = [.2, .3, .5]; expected = 0.
            for indices in itertools.product(range(3), repeat=4):
                expected += max(gains[i] for i in indices)*np.prod([p[i] for i in indices])
            actual = expected_best_iid(torch.tensor(gains, dtype=torch.float64), torch.tensor(p, dtype=torch.float64))
            self.assertAlmostEqual(float(actual), expected, places=14)

    def test_small_positive_reference_no_big_epsilon(self):
        logits = torch.tensor([[0., 1.]], requires_grad=True)
        gains = torch.tensor([[1e-25, 2e-25]], dtype=torch.float64); noop = torch.tensor(-1., requires_grad=True)
        terms = action_loss(logits, noop, gains, torch.ones_like(gains, dtype=torch.bool),
                            torch.zeros_like(gains), 2e-25, torch.tensor([[0, 1]]), torch.ones_like(gains, dtype=torch.bool))
        self.assertLess(float(terms["cover"].detach()), .2); self.assertTrue(torch.isfinite(terms["loss"]))
        terms["loss"].backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_reference_no_fake_ratio(self):
        gains = torch.tensor([[-2., 0.]], dtype=torch.float64)
        terms = action_loss(torch.zeros((1, 2)), torch.tensor(1.), gains, torch.ones((1, 2), dtype=torch.bool),
                            torch.zeros_like(gains), 0., torch.tensor([[0, 1]]), torch.ones((1, 2), dtype=torch.bool))
        self.assertEqual(float(terms["cover"]), 0.); self.assertTrue(terms["reference_zero"])

    def test_duplicate_singletons_not_penalized(self):
        g = torch.ones((1, 3), dtype=torch.float64); mask = torch.ones((1, 3), dtype=torch.bool)
        args = (torch.zeros((1, 3)), torch.tensor(0.), g, mask, g*0, 1.)
        singleton = action_loss(*args, torch.tensor([[0, 1, 2]]), mask)
        duplicate = action_loss(*args, torch.tensor([[0, 0, 2]]), mask)
        self.assertEqual(float(singleton["duplicate"]), 0.)
        self.assertAlmostEqual(float(duplicate["duplicate"]), .25)

    def test_infeasible_below_tau_and_incomplete(self):
        g = torch.tensor([[10., .2]], dtype=torch.float64)
        terms = action_loss(torch.zeros((1, 2)), torch.tensor(0.), g, torch.tensor([[False, True]]),
                            torch.tensor([[0., .3]]), 0., torch.tensor([[0, 1]]), torch.ones((1, 2), dtype=torch.bool))
        self.assertTrue(terms["reference_zero"])
        with self.assertRaises(ContractError): LabelBatch(g.numpy(), [[True, True]], [[0., 0.]], 10., [[0, 1]], complete=False)

    def test_stable_top4_noop_and_masks(self):
        f = features(3, 2); scores = np.zeros((3, 2))
        self.assertEqual(select_actions(f, scores, 0.), [])
        actions = select_actions(f, scores, -1.)
        self.assertEqual(len(actions), 4); self.assertEqual(len({a.action_id for a in actions}), 4)
        self.assertEqual([(a.recipe_id, a.drop_index) for a in actions],
                         [("A1:fourier:000", 0), ("A1:fourier:000", 1), ("A1:fourier:001", 0), ("A1:fourier:001", 1)])
        masked = replace(f, outgoing_mask=np.array([False, True]))
        self.assertTrue(all(a.drop_index == 1 for a in select_actions(masked, scores, -1)))

    def test_model_parameter_budget_padding_and_backward(self):
        model = ActionMLP(); self.assertGreaterEqual(model.parameter_count, 30000); self.assertLessEqual(model.parameter_count, 80000)
        xi = torch.zeros((1, 204)); nodes = torch.zeros((1, 3, 32)); outgoing = torch.zeros((1, 2, 32))
        nm = torch.tensor([[True, False, False]]); om = torch.tensor([[True, False]])
        logits, noop = model(xi, nodes, outgoing, nm, om)
        self.assertTrue(torch.isneginf(logits[0, 1:]).all()); self.assertTrue(torch.isfinite(noop).all())
        g = torch.ones((3, 2), dtype=torch.float64); legal = nm[0, :, None] & om[0, None, :]
        loss = action_loss(logits[0], noop[0], g, legal, g*0, 1., torch.arange(6).reshape(3, 2), legal)["loss"]
        loss.backward(); self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_all_padded_model_noop(self):
        model = ActionMLP()
        logits, noop = model(torch.zeros((1, 204)), torch.zeros((1, 0, 32)), torch.zeros((1, 2, 32)),
                             torch.zeros((1, 0), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (1, 0, 2)); self.assertTrue(torch.isfinite(noop).all())

    def test_no_reference_excluded_and_missing_kept(self):
        rows = [dict(object_id="a", family="near", gain=0., reference=0.),
                dict(object_id="b", family="near", gain=1., reference=2.),
                dict(object_id="c", family="near", gain=0., reference=2., missing=True)]
        s = coverage_summary(rows); self.assertEqual(s["macro"], .5)
        self.assertEqual(s["zero_reference_states"], 1); self.assertEqual(s["missing_states"], 1)
        self.assertIsNone(s["objects"][0]["coverage"])
        report = paired_bootstrap_objects(rows, rows)
        self.assertEqual(report["n_objects"], 1); self.assertEqual(report["interval95"], [0., 0.])

    def test_preflight_missing_timing_is_hold(self):
        f = features(split="development"); labels = LabelBatch(np.ones((3, 2)), np.ones((3, 2), bool), np.zeros((3, 2)), 1., np.arange(6).reshape(3, 2))
        item = DatasetItem(f, labels, "near", {"original_dictionary_gain_fp64": 1.})
        result, _ = assess([item], [])
        self.assertEqual(result["gates"]["G1A"], "HOLD"); self.assertFalse(result["training_authorized"])

    def test_training_authorisation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/"config.json").write_text("{}")
            (p/"gate.json").write_text(json.dumps({"gates": {"G0": "PASS", "G1A": "NOT_RUN", "G2": "PASS"}}))
            (p/"owner.json").write_text("{}")
            with self.assertRaises(ContractError): authorize(p/"config.json", p/"gate.json", p/"owner.json", "J1")


if __name__ == "__main__": unittest.main()
