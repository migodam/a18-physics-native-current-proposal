"""Publish this clean local Git repository using the user's authenticated gh CLI.

This helper is not run by the delivered research package. It requires explicit
owner/name, visibility (private by default), and --confirm. No force-push.
Official reference: https://cli.github.com/manual/gh_repo_create
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          capture_output=capture)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--visibility", choices=["private", "public"], default="private")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("No network write without --confirm")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("Expected owner/repository")
    for executable in ("git", "gh"):
        if not shutil.which(executable):
            parser.error(f"Required executable is missing: {executable}")
    run(["gh", "auth", "status"])
    if run(["git", "status", "--porcelain"], capture=True).stdout.strip():
        parser.error("Working tree is not clean. Review and commit changes first.")
    remotes = run(["git", "remote"], capture=True).stdout.splitlines()
    if "origin" in remotes:
        parser.error("origin already exists; refusing to overwrite or redirect it")
    run(["gh", "repo", "create", args.repo, "--source", str(ROOT),
         "--remote", "origin", "--push", "--" + args.visibility])
    url = run(["git", "remote", "get-url", "origin"], capture=True).stdout.strip()
    commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    remote = run(["git", "ls-remote", "origin", "refs/heads/main"], capture=True).stdout
    if commit not in remote:
        raise RuntimeError("Remote main commit does not match local HEAD; inspect publication manually")
    receipt = {"published": True, "repository": args.repo, "remote": url,
               "commit": commit, "visibility": args.visibility,
               "verified_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "results" / "publication_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
