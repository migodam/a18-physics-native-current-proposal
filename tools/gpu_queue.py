"""Cooperative, serial GPU-job wrapper. No training code is provided here.

Every job on one physical GPU must use the SAME absolute lock and receipt paths.
Occupied wall time, including failed/time-limited attempts, consumes the budget.
This cannot detect or stop unrelated programs that do not honor this lock.
"""
from __future__ import annotations
import argparse, json, os, signal, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

def read_receipts(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)  # Corrupt accounting fails closed.
            if row.get("occupied_seconds", -1) < 0:
                raise ValueError("Invalid receipt: occupied_seconds is missing/negative")
            rows.append(row)
    return rows

def stop_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=15)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pilot.json")
    parser.add_argument("--job", required=True)
    parser.add_argument("--receipt-dir", default="results/gpu_jobs")
    parser.add_argument("--lock-dir", default=".gpu.lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        parser.error("Specify a foreground command after --")
    config = json.loads(Path(args.config).read_text())
    if args.job not in config["jobs"]:
        parser.error("Unknown or unauthorized job ID; Route C has no job")
    job = config["jobs"][args.job]
    bucket = job["bucket"]
    caps = {"nn": config["training"]["total_nn_gpu_minutes_cap"],
            "physics": config["data"]["teacher_gpu_minutes_cap"],
            "teacher": config["data"]["teacher_cpu_minutes_cap"],
            "nonlinear": config["evaluation"]["nonlinear_gpu_minutes_cap"]}
    lock = Path(args.lock_dir).resolve()
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        parser.exit(2, f"GPU lock exists: {lock}. Do not auto-delete it.\n")
    receipt_dir = Path(args.receipt_dir).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ledger = receipt_dir / "jobs.jsonl"
    proc = None
    start = None
    status, returncode = "NOT_STARTED", 2
    try:
        (lock / "owner.json").write_text(json.dumps({
            "pid": os.getpid(), "host": socket.gethostname(), "job": args.job,
            "utc": datetime.now(timezone.utc).isoformat()}, indent=2))
        rows = read_receipts(ledger)
        used_job = sum(r["occupied_seconds"] for r in rows if r["job"] == args.job)
        used_bucket = sum(r["occupied_seconds"] for r in rows if r["bucket"] == bucket)
        allowed = min(60 * job["minutes_cap"] - used_job, 60 * caps[bucket] - used_bucket)
        if allowed <= 0:
            raise RuntimeError("Job or total bucket budget exhausted; no new process started")
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        popen_kwargs = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, **popen_kwargs)
        try:
            returncode = proc.wait(timeout=allowed)
            status = "COMPLETED" if returncode == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            status, returncode = "TIME_CAP", 124
            stop_tree(proc)
        except KeyboardInterrupt:
            status, returncode = "INTERRUPTED", 130
            stop_tree(proc)
    except Exception as exc:
        if proc is not None:
            stop_tree(proc)
        status = "WRAPPER_ERROR"
        print(f"GPU queue: {exc}", file=sys.stderr)
    finally:
        if start is not None:
            row = {"job": args.job, "bucket": bucket, "status": status,
                   "returncode": returncode, "occupied_seconds": time.perf_counter() - start,
                   "gpu_occupied_seconds": 0. if bucket == "teacher" else time.perf_counter() - start,
                   "utc": datetime.now(timezone.utc).isoformat(), "command": cmd,
                   "note": "Occupied wall, not measured GPU kernel utilization; physics gates authorized by integrator"}
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        owner = lock / "owner.json"
        if owner.exists():
            owner.unlink()
        lock.rmdir()
    return returncode

if __name__ == "__main__":
    raise SystemExit(main())
