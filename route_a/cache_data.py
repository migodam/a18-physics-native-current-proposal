from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from .audit import assert_online_origin, receipt_dict
from .recipes import RECIPE_SCHEMA_HASH
from .types import ContractError, DatasetItem, FeatureBatch, LabelBatch, SCHEMA_HASH, canonical_hash

IDENTITY = ("object_id", "state_id", "configuration_id", "backend_commit",
            "material_gauge_sha256", "receiver_geometry_sha256", "state_sha256")
ARRAY_KEYS = {"xi", "nodes", "outgoing", "node_mask", "outgoing_mask", "recipe_ids", "drop_indices", "metadata_json"}


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set(); objects = {}
    for row in rows:
        if any(not row.get(k) for k in IDENTITY) or not row.get("split"):
            raise ContractError("Unresolved frozen state identity")
        key = (row["state_id"], row["configuration_id"])
        if key in seen: raise ContractError("Duplicate state/configuration")
        seen.add(key)
        if row["object_id"] in objects and objects[row["object_id"]] != row["split"]:
            raise ContractError("Parent-object split leakage")
        objects[row["object_id"]] = row["split"]
        if row.get("residual_sign") != "predicted_minus_observed" or row.get("n_illuminations") != 4:
            raise ContractError("This schema requires the frozen four-illumination sign/packing contract")
    return rows


def load_online_features(path, manifest_row):
    path = Path(path).resolve(); assert_online_origin(path)
    expected = manifest_row.get("online_features_sha256")
    if not expected or file_hash(path) != expected:
        raise ContractError("Online cache needs a verified immutable hash")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != ARRAY_KEYS:
            raise ContractError("Online cache must use the declared whitelist, not extra teacher arrays")
        arrays = {k: z[k].copy() for k in z.files}
    meta = json.loads(str(arrays.pop("metadata_json").item()))
    for key in IDENTITY:
        if meta.get(key) != manifest_row[key]:
            raise ContractError(f"Online cache identity mismatch: {key}")
    if meta.get("schema_hash") != SCHEMA_HASH or meta.get("split") != manifest_row["split"]:
        raise ContractError("Cache schema/split mismatch")
    receipts = meta.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ContractError("Online features need source/dependency/cost provenance")
    assert_online_origin(receipts)
    for receipt in receipts: receipt_dict(receipt)
    provenance = meta.get("provenance", [])
    expected_channels = {("xi", None, i) for i in range(204)}
    expected_channels |= {("nodes", j, i) for j in range(len(arrays["recipe_ids"])) for i in range(32)}
    expected_channels |= {("outgoing", j, i) for j in range(len(arrays["drop_indices"])) for i in range(32)}
    actual_channels = {(p["block"], p["row"], p["column"]) for p in provenance}
    if actual_channels != expected_channels or len(provenance) != len(expected_channels):
        raise ContractError("Every feature must have one source/dependency/cost-group entry")
    assert_online_origin(provenance)
    for p in provenance:
        refs = p.get("cost_receipt_indices", [])
        if not refs or any(not isinstance(j, int) or not 0 <= j < len(receipts) for j in refs):
            raise ContractError("Feature cost references do not resolve")
    return FeatureBatch(arrays["xi"], arrays["nodes"], arrays["outgoing"],
                         arrays["node_mask"], arrays["outgoing_mask"],
                         tuple(str(x) for x in arrays["recipe_ids"]),
                         tuple(int(x) for x in arrays["drop_indices"]), meta["configuration_id"],
                         meta["object_id"], meta["state_id"], meta["split"], tuple(receipts), provenance=tuple(provenance))


def save_online_features(path, f, manifest_row):
    path = Path(path); assert_online_origin(path)
    if (f.object_id, f.state_id, f.state_key, f.split) != tuple(manifest_row[k] for k in
            ("object_id", "state_id", "configuration_id", "split")):
        raise ContractError("Feature identity cannot be relabelled while saving")
    meta = {k: manifest_row[k] for k in IDENTITY}
    meta.update(schema_hash=f.schema_hash, split=f.split, receipts=list(f.receipts), provenance=list(f.provenance))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xi=f.xi, nodes=f.nodes, outgoing=f.outgoing,
                        node_mask=f.node_mask, outgoing_mask=f.outgoing_mask,
                        recipe_ids=np.array(f.recipe_ids), drop_indices=np.array(f.drop_indices),
                        metadata_json=json.dumps(meta, sort_keys=True))
    return file_hash(path)


def load_dataset(manifest, feature_dir, labels_path, split):
    manifest = Path(manifest); rows = read_manifest(manifest)
    if split not in {"train", "validation", "development", "locked_test", "test"}:
        raise ContractError("Unknown split")
    wanted = [r for r in rows if r["split"] == split]
    if not wanted: raise ContractError(f"No states for split {split}")
    allowed = {(r["state_id"], r["configuration_id"]) for r in wanted}
    records = {}
    with Path(labels_path).open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            key = (row.get("state_id"), row.get("configuration_id"))
            if key not in allowed: continue
            if not row.get("recipe_id", "").startswith("A1:"):
                raise ContractError("A0 labels cannot supervise A1 recipes")
            action_key = (row["recipe_id"], int(row["drop_index"]))
            if action_key in records.setdefault(key, {}):
                raise ContractError("Duplicate action label")
            if row.get("reference_type") not in {"exact_enumerated_dictionary_reference", "audited_pool_reference"}:
                raise ContractError("Action labels require an audited reference identity")
            if row.get("utility_space") != "full_material":
                raise ContractError("Material sketches cannot stand in for full finite utility")
            records[key][action_key] = row
    items = []; directory = Path(feature_dir).resolve()
    for row in wanted:
        if row.get("recipe_schema_hash") != RECIPE_SCHEMA_HASH:
            raise ContractError("Manifest recipe schema does not match this A1 implementation")
        name = row.get("online_feature_file")
        if not name: raise ContractError("Integrator must assign the feature-cache filename")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory): raise ContractError("Feature path escapes online whitelist")
        f = load_online_features(path, row)
        n, k = len(f.recipe_ids), len(f.drop_indices)
        g = np.zeros((n, k)); tau = np.zeros((n, k)); feasible = np.zeros((n, k), bool)
        groups = np.arange(n * k).reshape(n, k)
        reference = row.get("common_reference_gain_fp64")
        if reference is None: raise ContractError("Missing common detached gain reference")
        labels = records.get((row["state_id"], row["configuration_id"]), {})
        group_ids = {}
        for j, rid in enumerate(f.recipe_ids):
            for i, drop in enumerate(f.drop_indices):
                label = labels.get((rid, drop))
                if label is None:
                    if f.node_mask[j] and f.outgoing_mask[i]:
                        raise ContractError(f"Missing active action label: {rid}/{drop}")
                    continue
                for key in IDENTITY:
                    if label.get(key) != row[key]: raise ContractError(f"Teacher identity mismatch: {key}")
                if label.get("recipe_schema_hash") != row.get("recipe_schema_hash") or not row.get("recipe_schema_hash"):
                    raise ContractError("New recipes require new teacher labels")
                if not isinstance(label.get("feasible"), bool): raise ContractError("Label needs an explicit feasibility flag")
                if label["gain_fp64"] is None or label["tau_online"] is None:
                    if label["feasible"] or not label.get("invalid_reason"):
                        raise ContractError("Undefined gain requires a completed invalid endpoint record")
                g[j, i] = np.nan if label["gain_fp64"] is None else label["gain_fp64"]
                tau[j, i] = np.nan if label["tau_online"] is None else label["tau_online"]
                feasible[j, i] = label["feasible"]
                canonical = str(label.get("canonical_group_id", f"{rid}|{drop}"))
                if canonical not in group_ids: group_ids[canonical] = len(group_ids)
                groups[j, i] = group_ids[canonical]
        items.append(DatasetItem(f, LabelBatch(g, feasible, tau, float(reference), groups), row.get("family", "unknown"), row))
    return items


def dataset_fingerprint(items):
    return canonical_hash([i.manifest for i in items])
