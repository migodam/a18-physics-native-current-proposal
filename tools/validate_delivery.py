"""Check local document links, syntax, dimensions, budgets, and honest status."""
from __future__ import annotations
import json, re
from pathlib import Path
from urllib.parse import unquote
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["START_HERE.md", "THEORY_OVERVIEW.md", "ROUTE_A_DICTIONARY_PROPOSAL.md",
            "ROUTE_B_ORTHOGONAL_FEEDBACK_PROPOSAL.md", "ROUTE_C_PROBABILISTIC_PRIOR.md",
            "ORTHOGONAL_POLYNOMIAL_THEORY.md", "NONNORMAL_MAXWELL_ANALYSIS.md",
            "NOVELTY_AUDIT.md", "SHARED_DATA_AND_TEACHER_SPEC.md",
            "LIGHTWEIGHT_EXPERIMENT_PLAN.md", "GPU_JOB_ORDER.md", "GO_NO_GO_GATES.md",
            "CODEX_ROUTE_A.md", "CODEX_ROUTE_B.md"]

def main():
    errors, checked_links = [], 0
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append("Missing: " + name)
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text()
        if sum(line.startswith("```") for line in text.splitlines()) % 2:
            errors.append("Unbalanced code fence: " + str(path.relative_to(ROOT)))
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                errors.append(f"Broken local link: {path.name} -> {target}")
            checked_links += 1
    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text())
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            compile(path.read_text(), str(path), "exec")
    config = json.loads((ROOT / "config/pilot.json").read_text())
    schema = json.loads((ROOT / "config/feature_schema.json").read_text())
    if sum(x["real_count"] for x in schema["B_blocks"]) != 1552:
        errors.append("B feature dimension mismatch")
    if sum(config["route_A"]["recipes"].values()) != 122:
        errors.append("Recipe count mismatch")
    if sum(j["minutes_cap"] for j in config["jobs"].values() if j["bucket"] == "nn") != 240:
        errors.append("NN job cap mismatch")
    if config["route_C"]["training_enabled"]:
        errors.append("Route C was enabled without an amended gate")
    algebra = json.loads((ROOT / "results/algebra_test_report.json").read_text())
    if not algebra["passed"] or algebra["tests_run"] != 17:
        errors.append("Expected 17 passing dense algebra tests")
    report = {"passed": not errors, "errors": errors,
              "required_documents_present": all((ROOT / n).is_file() for n in REQUIRED),
              "local_links_checked": checked_links, "feature_dimension_B": 1552,
              "recipe_cap": 122, "nn_job_cap_minutes": 240,
              "scope": "delivery consistency, not scientific pilot verification"}
    (ROOT / "results/DELIVERY_QA.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
