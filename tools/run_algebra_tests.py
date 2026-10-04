"""Run the independent dense algebra suite and save an honest CPU-only receipt."""
from __future__ import annotations
import io, json, platform, sys, time, unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from ids(item)
        else:
            yield item.id()

def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_algebra.py")
    names = list(ids(suite))
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    report = {
        "scope": "synthetic_dense_CPU_algebra_only",
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "passed": result.wasSuccessful(), "elapsed_seconds": elapsed,
        "test_ids": names,
        "maxwell_backend_replayed": False, "neural_network_trained": False,
        "gpu_used": False, "deterministic_safety_certificate": False,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "algebra_test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "algebra_test_output.txt").write_text(stream.getvalue())
    print(stream.getvalue(), end="")
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
