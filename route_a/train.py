from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
import os
import random
import socket
import time
import numpy as np
import torch
from .audit import cached_epoch
from .baselines import analytic_actions, all_registered_policies
from .cache_data import load_dataset, dataset_fingerprint, file_hash
from .features import Normalizer
from .loss import action_loss
from .metrics import coverage_summary, selected_gain
from .model import ActionMLP
from .types import ContractError, SCHEMA_HASH


def pack_batch(items, device):
    batch = len(items); n = max(len(i.features.recipe_ids) for i in items)
    k = max(len(i.features.drop_indices) for i in items)
    xi = np.stack([i.features.xi for i in items]); nodes = np.zeros((batch, n, 32))
    outgoing = np.zeros((batch, k, 32)); nm = np.zeros((batch, n), bool); om = np.zeros((batch, k), bool)
    gain = np.zeros((batch, n, k)); tau = np.zeros_like(gain); feasible = np.zeros_like(gain, bool)
    groups = np.arange(batch * n * k).reshape(batch, n, k)
    refs = np.array([i.labels.reference_fp64 for i in items])
    for b, item in enumerate(items):
        f, l = item.features, item.labels; nn, kk = len(f.recipe_ids), len(f.drop_indices)
        nodes[b, :nn] = f.nodes; outgoing[b, :kk] = f.outgoing
        nm[b, :nn] = f.node_mask; om[b, :kk] = f.outgoing_mask
        gain[b, :nn, :kk] = l.gains_fp64; tau[b, :nn, :kk] = l.tau_online
        feasible[b, :nn, :kk] = l.feasible; groups[b, :nn, :kk] = l.canonical_groups
    arrays = (xi, nodes, outgoing, nm, om, gain, feasible, tau, refs, groups)
    dtypes = (torch.float32, torch.float32, torch.float32, torch.bool, torch.bool,
              torch.float64, torch.bool, torch.float64, torch.float64, torch.int64)
    return [torch.as_tensor(a, device=device, dtype=t) for a, t in zip(arrays, dtypes)]


def cached_policy_rows(model, raw_items, normalizer, K=4):
    rows = []; model.eval()
    for item in raw_items:
        actions = model.propose(normalizer.transform(item.features), K)
        rows.append({"object_id": item.features.object_id, "state_id": item.features.state_id,
                     "configuration_id": item.features.state_key, "family": item.family,
                     "gain": selected_gain(item, actions), "reference": item.labels.reference_fp64,
                     "selected_actions": len(actions), "source": "cached_action_labels",
                     "T_acq": None, "T_inc": None})
    return rows


def choose_analytic(items):
    best = None
    for policy in all_registered_policies():
        rows = []
        for item in items:
            actions = analytic_actions(item.features, **policy)
            rows.append({"object_id": item.features.object_id, "family": item.family,
                         "gain": selected_gain(item, actions), "reference": item.labels.reference_fp64})
        metrics = coverage_summary(rows); cov = metrics["macro"]
        if cov is not None and (best is None or cov > best["coverage"]):
            best = {"policy": policy, "coverage": cov, "metrics": metrics}
    return best


def train_cached(train_items, validation_items, config, out, seed=1801, device="cpu"):
    """No backend capability is available to this function.

    CPU test fixtures may call it directly; the CLI separately requires real
    G0/G1A/G2 and a live Integrator queue owner receipt.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = config["route_A"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    normalizer = Normalizer.fit([i.features for i in train_items])
    training = [replace(i, features=normalizer.transform(i.features)) for i in train_items]
    baseline = choose_analytic(validation_items)
    if baseline is None: raise ContractError("Validation lacks positive complete object references")
    model = ActionMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = np.random.default_rng(seed); best = -np.inf; last_improvement = 0
    history = []; reason = "MAX_EPOCHS"; started = time.perf_counter(); checkpoint = None
    for epoch in range(1, cfg["epochs_max"] + 1):
        if time.perf_counter()-started >= 60 * config["jobs"]["J1"]["minutes_cap"]:
            reason = "TIME_CAP"; break
        model.train(); order = generator.permutation(len(training)); losses = []
        with cached_epoch() as physics:
            for begin in range(0, len(order), cfg["batch_states"]):
                items = [training[int(j)] for j in order[begin:begin + cfg["batch_states"]]]
                tensors = pack_batch(items, device)
                xi, nodes, outgoing, nm, om, gain, feasible, tau, refs, groups = tensors
                logits, noop = model(xi, nodes, outgoing, nm, om)
                batch_losses = [action_loss(logits[b], noop[b], gain[b], feasible[b], tau[b], refs[b], groups[b],
                                            nm[b, :, None] & om[b, None, :])["loss"] for b in range(len(items))]
                loss = torch.stack(batch_losses).mean()
                if not torch.isfinite(loss): raise ContractError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
                optimizer.step(); losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **physics}
        if epoch % config["training"]["validation_every_epochs"] == 0:
            rows = cached_policy_rows(model, validation_items, normalizer)
            metrics = coverage_summary(rows); cov = metrics["macro"]
            record["validation_macro"] = cov; record["validation_near_contact"] = metrics["near_contact"]
            if cov is not None and cov > best:
                best = cov; last_improvement = epoch
                checkpoint = {"weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "normalizer": normalizer.as_dict(), "normalizer_hash": normalizer.fingerprint,
                              "schema_hash": SCHEMA_HASH, "config": config, "seed": seed, "epoch": epoch,
                              "parameters": model.parameter_count, "train_hash": dataset_fingerprint(train_items),
                              "validation_hash": dataset_fingerprint(validation_items), "analytic_baseline": baseline}
                torch.save(checkpoint, out / "best.pt")
        history.append(record)
        with (out / "epochs.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
        if epoch >= cfg["check_epoch"] and best <= baseline["coverage"]:
            reason = "NO_POSITIVE_VALIDATION_HEADROOM"; break
        if epoch >= cfg["check_epoch"] and epoch-last_improvement >= cfg["patience_epochs"]:
            reason = "PATIENCE"; break
    if checkpoint is not None: model.load_state_dict(checkpoint["weights"])
    result = {"status": "CACHED_TRAINING_COMPLETE_ONLINE_COST_PENDING", "reason": reason,
              "epochs": len(history), "parameters": model.parameter_count, "seed": seed,
              "best_validation_macro": None if not np.isfinite(best) else best,
              "analytic_validation_macro": baseline["coverage"], "physics_actions_in_epochs": 0,
              "maxwell_online_timing": "NOT_MEASURED", "G3": "NOT_RUN",
              "elapsed_seconds": time.perf_counter()-started, "checkpoint_written": checkpoint is not None}
    (out / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return model, normalizer, result


def authorize(config_path, gate_path, queue_path, job):
    gate = json.loads(Path(gate_path).read_text()); owner = json.loads(Path(queue_path).read_text())
    for name in ("G0", "G1A", "G2"):
        status = gate.get("gates", {}).get(name)
        if isinstance(status, dict): status = status.get("status")
        if status != "PASS": raise ContractError(f"Training requires {name}=PASS")
    if gate.get("config_sha256") != file_hash(config_path) or gate.get("evidence_scope") != "real_vector_maxwell":
        raise ContractError("Gate authorization must bind actual config and real Maxwell evidence")
    path = Path(queue_path)
    if owner.get("job") != job or owner.get("host") != socket.gethostname() or not owner.get("pid"):
        raise ContractError("Wrong Integrator job/host owner receipt")
    if path.name != "owner.json" or path.parent.name not in {"gpu.lock", ".gpu.lock"}:
        raise ContractError("Training must run inside the shared Integrator queue")
    return gate


def main():
    parser = argparse.ArgumentParser(description="Integrator-authorized cached A1-MLP training")
    parser.add_argument("--config", default="config/pilot.json"); parser.add_argument("--job", choices=["J1"], required=True)
    parser.add_argument("--seed", type=int, default=1801); parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--features", default="cache/online_features"); parser.add_argument("--labels", default="labels/action_A1.jsonl")
    parser.add_argument("--gate-receipt", default="results/route_a/GATE_DECISION.json")
    parser.add_argument("--queue-owner", default=os.environ.get("A18_QUEUE_OWNER", ""))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="results/route_a/J1")
    args = parser.parse_args()
    authorize(args.config, args.gate_receipt, args.queue_owner, args.job)
    cfg = json.loads(Path(args.config).read_text())
    if args.seed != cfg["training"]["first_seed"]:
        raise ContractError("Additional seeds belong to the Integrator's winner job")
    training = load_dataset(args.manifest, args.features, args.labels, "train")
    validation = load_dataset(args.manifest, args.features, args.labels, "validation")
    if {i.features.object_id for i in training} & {i.features.object_id for i in validation}:
        raise ContractError("Train/validation object leakage")
    _, _, result = train_cached(training, validation, cfg, args.output, args.seed, args.device)
    print(json.dumps(result))


if __name__ == "__main__": main()
