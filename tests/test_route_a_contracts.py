"""Synthetic CPU contract tests; do not constitute G0 or Maxwell evidence."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from test_route_a_acquisition import fixture
from test_route_a_learning import features
from route_a.audit import AcquisitionAudit, charged_apply
from route_a.cache_data import save_online_features, load_online_features, load_dataset, file_hash, read_manifest
from route_a.features import extract, Normalizer
from route_a.loss import action_loss, expected_best_iid
from route_a.recipes import RECIPE_SCHEMA_HASH, LazyCache, feedback_apply, InvalidCurrent
from route_a.runtime import RouteAPlugin, verify_policy, CheckpointPolicy
from route_a.types import ContractError, LabelBatch, DatasetItem, action, SCHEMA_HASH
from route_a.bridge import ReceiptBackend
from route_a.model import ActionMLP
from route_a.baselines import analytic_actions
from route_a.preflight import assess
from src.algebra import galerkin_complex, realify_jacobians, realify_data, frozen_step, finite_gain
from shared_backend import OperatorBackend


def cache_fixture(directory):
    s, b, _ = fixture(); f, _, _ = extract(s, b)
    row = dict(object_id=f.object_id, state_id=f.state_id, configuration_id=f.state_key, split=f.split,
               backend_commit="d"*40, material_gauge_sha256="b"*64, receiver_geometry_sha256="c"*64,
               state_sha256="a"*64, residual_sign="predicted_minus_observed", n_illuminations=4,
               family="near", online_feature_file="one.npz", recipe_schema_hash=RECIPE_SCHEMA_HASH,
               common_reference_gain_fp64=1.)
    row["online_features_sha256"] = save_online_features(directory/"one.npz", f, row)
    labels = []
    for j, rid in enumerate(f.recipe_ids):
        for i, drop in enumerate(f.drop_indices):
            if not f.node_mask[j] or not f.outgoing_mask[i]: continue
            labels.append(dict(row, recipe_id=rid, drop_index=drop, gain_fp64=.2, tau_online=1e-12,
                               feasible=True, reference_type="audited_pool_reference", utility_space="full_material"))
    return f, row, labels


class ContractTests(unittest.TestCase):
    def test_online_cache_roundtrip_and_extra_teacher_array_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); f, row, _ = cache_fixture(p)
            loaded = load_online_features(p/"one.npz", row)
            np.testing.assert_array_equal(loaded.nodes, f.nodes)
            self.assertEqual(len(loaded.provenance), 204+len(f.nodes)*32+len(f.outgoing)*32)
            with np.load(p/"one.npz", allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
            arrays["teacher_labels"] = np.zeros(1); np.savez_compressed(p/"one.npz", **arrays)
            row["online_features_sha256"] = file_hash(p/"one.npz")
            with self.assertRaises(ContractError): load_online_features(p/"one.npz", row)

    def test_a0_and_missing_labels_rejected_completed_invalid_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); f, row, labels = cache_fixture(p)
            manifest=p/"manifest.jsonl"; target=p/"labels.jsonl"
            manifest.write_text(json.dumps(row)+"\n")
            def write(rows): target.write_text("".join(json.dumps(r)+"\n" for r in rows))
            write(labels[:-1])
            with self.assertRaises(ContractError): load_dataset(manifest,p,target,"train")
            bad=[dict(r) for r in labels];bad[0]["recipe_id"]="A0:fourier:000";write(bad)
            with self.assertRaises(ContractError): load_dataset(manifest,p,target,"train")
            labels[0].update(feasible=False, gain_fp64=None, tau_online=None, invalid_reason="endpoint_failed")
            write(labels); item=load_dataset(manifest,p,target,"train")[0]
            self.assertEqual(int(np.isnan(item.labels.gains_fp64).sum()),1)
            self.assertTrue(np.isfinite(item.labels.opportunity()).all())

    def test_split_leakage_and_cache_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); f,row,_=cache_fixture(p)
            bad=dict(row,state_id="other-state",split="test")
            (p/"manifest.jsonl").write_text(json.dumps(row)+"\n"+json.dumps(bad)+"\n")
            with self.assertRaises(ContractError): read_manifest(p/"manifest.jsonl")
            with self.assertRaises(ContractError): load_online_features(p/"one.npz",dict(row,configuration_id="stale"))

    def test_zero_reference_only_noop_and_empty_bank_loss(self):
        g=torch.zeros((1,2),dtype=torch.float64);legal=torch.ones_like(g,dtype=torch.bool)
        terms=action_loss(torch.zeros((1,2)),torch.tensor(0.),g,legal,g,0.,torch.zeros_like(g,dtype=torch.long),legal)
        self.assertEqual(float(terms["duplicate"]),0.)
        self.assertAlmostEqual(float(terms["loss"]),np.log(3))
        z=torch.zeros((0,2));mask=torch.zeros((0,2),dtype=torch.bool)
        terms=action_loss(z,torch.tensor(0.,requires_grad=True),z.double(),mask,z.double(),0.,z.long(),mask)
        self.assertEqual(float(terms["loss"].detach()),0.)

    def test_tiny_probability_and_float64_division_floor(self):
        expected=expected_best_iid(torch.tensor([0.,1.],dtype=torch.float64),torch.tensor([1.,1e-40],dtype=torch.float64))
        self.assertAlmostEqual(float(expected)/1e-40,4.)
        g=torch.tensor([[1e-320]],dtype=torch.float64);mask=torch.ones_like(g,dtype=torch.bool)
        terms=action_loss(torch.tensor([[0.]],requires_grad=True),torch.tensor(0.),g,mask,g*0,1e-320,g.long(),mask)
        self.assertTrue(terms["division_floor_active"]);self.assertTrue(torch.isfinite(terms["loss"]))
        terms["loss"].backward()

    def test_explicit_invalid_label_does_not_poison_loss(self):
        g=torch.tensor([[float("nan"),1.]],dtype=torch.float64);valid=torch.tensor([[False,True]])
        terms=action_loss(torch.zeros((1,2),requires_grad=True),torch.tensor(0.),g,valid,g*0,1.,torch.tensor([[0,1]]),torch.ones_like(valid))
        self.assertTrue(torch.isfinite(terms["loss"]));terms["loss"].backward()
        with self.assertRaises(ContractError): LabelBatch(g.numpy(),[[True,True]],[[0.,0.]],1.,[[0,1]])

    def test_endpoint_failure_and_duplicate_action_no_retry_or_ridge(self):
        s,b,_=fixture();plugin=RouteAPlugin(b);f,_=plugin.prepare(s)
        a=action(next(r for r in f.recipe_ids if ":fourier:" in r),1,s.configuration_id,0.)
        class Failed:
            calls=0
            def evaluate(self,a):
                self.calls+=1
                return dict(status="INVALID_ENDPOINT",reason="CURRENT_NEAR_SINGULAR; no jitter",cost={})
        verifier=Failed();row=verify_policy(plugin,s,[a],verifier)
        self.assertTrue(row["no_op"]);self.assertEqual(verifier.calls,1)
        self.assertEqual(len(row["attempts"]),1)
        with self.assertRaises(ContractError): verify_policy(plugin,s,[a,replace(a,action_id="fake-new-id")],verifier)
        self.assertEqual(verifier.calls,1)
        s.D[:]=0
        with self.assertRaises(InvalidCurrent): feedback_apply(np.ones(24),s,b,AcquisitionAudit())

    def test_recipe_phase_and_receiver_permutation_same_endpoint_full_gain(self):
        s,b,L=fixture();plugin=RouteAPlugin(b);f,_=plugin.prepare(s)
        rid=next(r for r in f.recipe_ids if ":fourier:" in r);a=action(rid,1,s.configuration_id,0.)
        c=plugin.instantiate(a);retained=np.delete(s.U,1,axis=1);W=np.column_stack((retained,c))
        rng=np.random.default_rng(18);S=rng.normal(size=(7,24))+1j*rng.normal(size=(7,24))
        Bs=[rng.normal(size=(24,4))+1j*rng.normal(size=(24,4)) for _ in range(4)]
        rs=[rng.normal(size=7)+1j*rng.normal(size=7) for _ in range(4)]
        Lambda=.3*np.eye(4);ell=rng.normal(size=4);old=np.zeros(4)
        def evaluate(W,SS,rr):
            reduced=realify_jacobians([galerkin_complex(L,SS,Bi,W) for Bi in Bs])
            J=realify_jacobians([SS@np.linalg.solve(L,Bi) for Bi in Bs]);r=realify_data(rr)
            step=frozen_step(reduced,r,Lambda,ell)
            return step,finite_gain(J,r,Lambda,ell,old,step)
        base=evaluate(W,S,rs);rot=W.copy();rot[:,-1]*=np.exp(.8j)
        perm=rng.permutation(7);other=evaluate(rot,S[perm],[r[perm] for r in rs])
        np.testing.assert_allclose(base[0],other[0],atol=1e-12);self.assertAlmostEqual(base[1],other[1],places=10)

    def test_missing_machinery_counts_cannot_pass_preflight(self):
        f=features(split="development");l=LabelBatch(np.ones((3,2)),np.ones((3,2),bool),np.zeros((3,2)),1.,np.arange(6).reshape(3,2))
        item=DatasetItem(f,l,"near",dict(original_dictionary_gain_fp64=1.,evidence_scope="real_vector_maxwell"))
        timing=dict(state_id=f.state_id,configuration_id=f.state_key,scope="real_online_matched_timing",complete=True,
                    source_sha256="synthetic-no-science",config_sha256="same",T_acq_Dfull=1.,T_acq_A1=.1,A1_counters={})
        gate=dict(gates=dict(G0="PASS"),evidence_scope="real_vector_maxwell",config_sha256="same")
        r,_=assess([item],[timing],gate,"same")
        self.assertEqual(r["gates"]["G1A"],"HOLD");self.assertFalse(r["training_authorized"])

    def test_norm_scales_train_only_and_checkpoint_roundtrip(self):
        f=features();xi=f.xi.copy();xi[32]=np.log1p(7.);xi[192]=np.log1p(3.);f=replace(f,xi=xi)
        normalizer=Normalizer.fit([f]);restored=Normalizer.from_dict(normalizer.as_dict())
        self.assertEqual(restored.fingerprint,normalizer.fingerprint)
        np.testing.assert_allclose(restored.transform(f).xi,normalizer.transform(f).xi)
        model=ActionMLP()
        payload=dict(weights=model.state_dict(),normalizer=normalizer.as_dict(),normalizer_hash=normalizer.fingerprint,schema_hash=SCHEMA_HASH)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"untrained_synthetic.pt";torch.save(payload,p)
            policy=CheckpointPolicy(p,file_hash(p));self.assertLessEqual(len(policy.propose(f)),4)

    def test_receipt_rhs_not_double_counted_and_shared_bridge_contract(self):
        s,_,L=fixture();rng=np.random.default_rng(1801);p=4;m=7
        B=rng.normal(size=(24,4,p))+1j*rng.normal(size=(24,4,p));S=rng.normal(size=(m,24))+1j*rng.normal(size=(m,24))
        meta=dict(chi=s.chi,a=np.ones(8),points=s.points,grid_shape=(2,2,2),cell_volume=1.,wavenumber=1.,
                  r=rng.normal(size=(4,m))+1j*rng.normal(size=(4,m)),ell=np.ones(p),Lambda=np.eye(p),
                  forcing=rng.normal(size=(24,4))+1j*rng.normal(size=(24,4)),state_hash="synthetic",
                  material_gauge_sha256="b"*64,receiver_geometry_sha256="c"*64,geometry={})
        shared=OperatorBackend(lambda c:L@c,lambda c:L.conj().T@c,lambda c:S@c,lambda w:S.conj().T@w,
            lambda z:np.einsum("nip,p...->ni...",B,z),lambda w,i:np.zeros(p),
            lambda c:np.einsum("nk,nip->ikp",c.conj(),B),lambda z:np.zeros(4*2*m),lambda k:s.U[:,:k],meta)
        frozen=shared.freeze_state(s.U,object_id="synthetic",state_id="one",receiver_mode_ids=(0,1,2))
        wrapped=ReceiptBackend(shared);plugin=RouteAPlugin(wrapped);f,audit=plugin.prepare(frozen)
        self.assertEqual(f.xi.shape,(204,));self.assertEqual(audit.calls["raw_currents"],0)
        before=audit.rhs.get("L",0);charged_apply(wrapped,"apply_L",np.ones((24,2)),audit)
        self.assertEqual(audit.rhs["L"]-before,2)

    def test_analytic_pairs_four_unique_incomings_with_one_legal_drop(self):
        f=features(n=7,k=2);f=replace(f,outgoing_mask=np.array([False,True]))
        actions=analytic_actions(f)
        self.assertEqual(len(actions),4);self.assertEqual(len({a.recipe_id for a in actions}),4)
        self.assertTrue(all(a.drop_index==1 for a in actions))


if __name__=="__main__":unittest.main()
