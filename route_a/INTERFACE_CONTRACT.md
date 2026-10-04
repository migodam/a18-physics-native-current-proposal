# Route A interface and evidence contract

Integrator owns shared physics, manifests, teachers, queue and final publication. Route A owns lazy recipes, feature assembly, policy, tests and reports. No Route A module implements a Maxwell inverse or endpoint solver.

## Shared wiring

```python
from route_a.bridge import ReceiptBackend
from route_a.runtime import RouteAPlugin, verify_policy
from route_a.baselines import analytic_actions

# b is shared_backend.OperatorBackend; s is its FrozenState.
wire = ReceiptBackend(b, seed_vector_encoding="unit_directions_columns")
plugin = RouteAPlugin(wire)
features, audit = plugin.prepare(s)
actions = analytic_actions(features, K=4)
verifier = wire.verifier(plugin, s)
round_result = verify_policy(plugin, s, [a.to_shared() for a in actions], verifier)
```

Shared callbacks return arrays and mutate CostReceipt. The bridge returns `(array, exclusive receipt)`, preserving counters and RHS. SeedBundle is n×6 unit directions; raw seeds are reconstructed from norms. Inactive directions remain zero while their below-floor norm diagnostic is retained. Selected callbacks materialize only the raw current; the shared verifier owns projection/normalization. `plugin.instantiate` is available for separate projection checks.

Action fields match shared wire: `route='A', action_id, configuration_id, drop_indices=(i,), prior_logit, recipe_id`. No gain is predicted. Highest no-op including a tie returns []; otherwise stable score/recipe/drop ordering selects at most four legal unique pairs. A0 IDs cannot enter A1. Integrator accepts the verified winner and calls `prepare(new_state)` before another proposal; at most three acceptances, with protected slots preserved.

## Features and caches

FeatureBatch contains readonly xi204, nodes N×32, outgoing k×32, masks, A1 recipe IDs, actual drop IDs, configuration/object/state/split identities, receipts and one provenance row per channel. N≤122, k≤16. A provenance row references a measured exclusive cost group, charged once; available timing resolution is explicit. No atom–seed projections or full recipe current bank are generated for descriptors.

Cached state norm channels32,36..47,51,52,59..61,192..197 use log1p(raw_norm), scale1. Train-only normalization replays nonzero training medians then z-scores. Node norm21 and outgoing norms16/17 are cached raw and log-scaled in normalization. Masks, one-hots and flags keep their meaning. Exported hashes: `route_a.types.SCHEMA_HASH`, `route_a.recipes.RECIPE_SCHEMA_HASH`.

NPZ whitelist: `xi,nodes,outgoing,node_mask,outgoing_mask,recipe_ids,drop_indices,metadata_json`; no pickle/teacher arrays. Metadata binds object/state/configuration IDs, backend_commit, material_gauge_sha256, receiver_geometry_sha256, state_sha256, split, schema_hash, receipts and provenance. Manifest assigns immutable online_features_sha256, relative online_feature_file, recipe_schema_hash, four illuminations and residual_sign='predicted_minus_observed'. Parent objects cannot cross splits.

A1 JSONL labels bind every active recipe/drop pair to these identities/schema, with gain_fp64,tau_online,feasible,canonical_group_id,reference_type,utility_space='full_material'. Missing active labels fail closed. Completed invalid endpoints can have null gain/tau only with feasible=false and invalid_reason; undefined raw values remain offline NaN, not measured zero. Manifest common_reference_gain_fp64 is required. A0 replay retains separate IDs/provenance and never supervises A1.

## Gates and CLI

```text
python -m route_a.audit --config config/pilot.json --split development
python -m route_a.audit --state-npz <public_state.npz> --object-id train_00 --state-id state0 --output results/route_a/interface_train00.json
python -m route_a.preflight --g0-receipt <integrator_G0_authorisation.json> --manifest data/manifest.jsonl
python -m route_a.train --job J1 --queue-owner <absolute shared gpu.lock/owner.json> --gate-receipt <actual A gate.json>
python -m route_a.evaluate --checkpoint <frozen.pt> --split locked_test --frozen-manifest <freeze.json> --evaluation-authorisation <window.json>
```

Real-state audit is CPU interface smoke, not oracle coverage or deployment timing. Missing inputs produce HOLD. Real preflight needs Integrator G0 receipt gates.G0='PASS',evidence_scope='real_vector_maxwell',config_sha256; manifest rows carry the real evidence scope. Matched timing rows need state/config IDs, config/source hashes, scope='real_online_matched_timing',complete=true,T_acq_Dfull,T_acq_A1,A1_counters. Forbidden machinery and full_SVD counters must all be present; geometry SVD needs geometry_SVD_separately_charged=true. Missing is not zero. Original dictionary and common-union references are reported separately.

Training requires real G0/G1A/G2, cached train/validation records and live J1 shared queue owner. Integrator launches it; epoch physics acquisition is forbidden. Frozen evaluation binds checkpoint/manifest/normalizer, consumes shared split test for locked_test, rejects historical exposure and atomically claims the unified window. Cached coverage never satisfies deployment G3/nonlinear G8; Integrator measures those online.
