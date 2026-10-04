"""Render audited object coverage from CSV; no physics or teacher access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/route_a/preflight/T0_object_coverage.csv")
    parser.add_argument("--review", default="results/route_a/preflight/T0_CODEX_REVIEW.json")
    parser.add_argument("--output-prefix", default="results/route_a/preflight/T0_oracle_coverage")
    args = parser.parse_args()
    source, review_path, prefix = Path(args.input), Path(args.review), Path(args.output_prefix)
    with source.open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: (r["object_id"].startswith("test"), r["object_id"]))
    review = json.loads(review_path.read_text())
    y = np.arange(len(rows))
    a, d, original = [np.array([float(r[key]) for r in rows]) for key in
                       ("A_coverage_union", "D_coverage_union", "A_coverage_original")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, layout="constrained")
    ax[0].barh(y-.18, d, .34, label="Original dictionary", color="#b0b6bf")
    ax[0].barh(y+.18, a, .34, label="A1 bank oracle", color="#2a678d")
    ax[1].barh(y, original, .52, color="#378074")
    ax[0].set_yticks(y, [r["object_id"]+" ("+r["family"]+")" for r in rows])
    ax[0].invert_yaxis()
    for panel in ax:
        panel.set_xlim(0, max(1.08, float(max(a.max(), d.max(), original.max()))+.1))
        panel.grid(axis="x", alpha=.25); panel.set_axisbelow(True)
        panel.set_xlabel("Object coverage: sum gain / sum reference")
        panel.spines[["top", "right"]].set_visible(False)
    for i, (v, w) in enumerate(zip(a, d)):
        ax[0].text(v+.014, i+.18, f"{v:.3f}", va="center", fontsize=9)
        ax[0].text(w+.014, i-.18, f"{w:.3f}", va="center", fontsize=9, color="#555555")
    for i, v in enumerate(original):
        ax[1].text(v+.014, i, f"{v:.3f}", va="center", fontsize=9)
    for panel, key, title in zip(ax, ("oracle_union", "oracle_original_dictionary"),
                                 ("Common singleton union reference", "Original dictionary reference")):
        metric = review[key]
        panel.set_title(f"{title}\nA1 macro {metric['macro']:.4f}; near-contact {metric['near_contact']:.4f}", fontsize=11)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    count_states = sum(int(r["states"]) for r in rows)
    fig.suptitle(f"Route A representation preflight: {len(rows)} exposed objects / {count_states} states", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("png", "svg"):
        target = prefix.with_suffix("."+ext); fig.savefig(target, dpi=160)
        outputs[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    plt.close(fig)
    metadata = dict(scope="exposed development representation only; not NN or locked-test",
                    input=source.name, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    review=review_path.name, review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
                    aggregation=review["aggregation"], n_objects=len(rows),
                    n_states=sum(int(r["states"]) for r in rows), outputs=outputs)
    prefix.with_name(prefix.name+"_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
