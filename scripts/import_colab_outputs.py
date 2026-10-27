"""Copy the outputs of a Colab or molab run into the project, checking every file on the way.

Give the folder that was downloaded (the one holding outputs_manifest.json).
Each file is checked against its hash before and after copying. Every result file must
come from a commit of this repository with no uncommitted changes, and from the same cache
that was built here. Nothing is copied if a check fails, and an existing file with
different content is never overwritten silently.

Usage:
    uv run python scripts/import_colab_outputs.py "/mnt/c/Users/<you>/Downloads/outputs"
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from digikala_reviews.config import CACHE_DIR, PROJECT_ROOT
from digikala_reviews.ingest import file_sha256

ALLOWED_TOP_FOLDERS = {"reports", "artifacts"}
RESULT_FOLDERS = (
    "reports/tfidf_runs/",
    "reports/tfidf_experiments/",
    "reports/tfidf_extra/",
    "reports/tfidf_final/",
    "reports/tokenizers/",
    "reports/transformer_pilot/",
    "reports/transformer_pilot_smoke/",
)


def commit_known(commit):
    if not commit:
        return False
    check = ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"]
    return subprocess.run(check, capture_output=True).returncode == 0


def check_run_file(relative, path, local_cache_hashes):
    """A result JSON must name a known commit, clean code and the local cache files."""
    run = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if not commit_known(run["code"]["commit"]) or run["code"]["uncommitted_changes"] is not False:
        problems.append(f"{relative}: made from unknown or uncommitted code")
    for name, digest in run["data"]["sha256"].items():
        if local_cache_hashes.get(name) != digest:
            problems.append(f"{relative}: made from a different {name} than the local cache")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace local files whose content differs")
    args = parser.parse_args()

    manifest = json.loads((args.source / "outputs_manifest.json").read_text(encoding="utf-8"))
    commit = manifest["code"]["commit"]
    cache_info = json.loads((CACHE_DIR / "cache_info.json").read_text(encoding="utf-8"))
    local_cache_hashes = {name: info["sha256"] for name, info in cache_info["files"].items()}
    problems = [] if commit_known(commit) else [f"commit {commit} is not in this repository"]

    listed = set(manifest["files"]) | {"outputs_manifest.json"}
    for path in sorted(args.source.rglob("*")):
        relative = path.relative_to(args.source).as_posix()
        if path.is_file() and relative not in listed:
            print(f"  note: {relative} is not in the manifest and will not be copied")

    plan = []
    for relative, expected in manifest["files"].items():
        source, target = args.source / relative, PROJECT_ROOT / relative
        inside = target.resolve().is_relative_to(PROJECT_ROOT.resolve())
        if not inside or Path(relative).parts[0] not in ALLOWED_TOP_FOLDERS or relative.endswith(".tmp"):
            problems.append(f"{relative}: not a file that belongs in reports/ or artifacts/")
        elif not source.exists():
            problems.append(f"{relative}: missing from the downloaded folder")
        elif file_sha256(source) != expected:
            problems.append(f"{relative}: damaged in the download (hash differs)")
        elif relative.startswith(RESULT_FOLDERS) and (
            run_problems := check_run_file(relative, source, local_cache_hashes)
        ):
            problems += run_problems
        elif target.exists() and file_sha256(target) != expected and not args.overwrite:
            problems.append(f"{relative}: a different local file already exists (use --overwrite to replace)")
        else:
            plan.append((source, target, expected))

    if problems:
        print("Nothing was copied:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    for source, target, expected in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert file_sha256(target) == expected, f"{target} changed while copying"
    print(f"Copied and checked {len(plan)} files from commit {commit}")


if __name__ == "__main__":
    main()
