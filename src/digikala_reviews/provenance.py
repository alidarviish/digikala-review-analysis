"""Record where a result came from: the code version, the libraries and the input data.

Every result file carries this record, so each number in a report can be traced back
to one commit, one set of library versions and one exact data file.
"""

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from digikala_reviews.config import PROJECT_ROOT
from digikala_reviews.ingest import file_sha256

LIBRARIES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "transformers", "tokenizers"]


def git_commit(root: Path = PROJECT_ROOT) -> dict:
    """The commit of the code, and whether there were uncommitted changes."""

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "uncommitted_changes": bool(git("status", "--porcelain")),
    }


def library_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "libraries": {name: library_version(name) for name in LIBRARIES},
        "machine": platform.node(),
    }


def check_cache(cache_dir: Path, names: list[str]) -> dict:
    """Compare the cached Parquet files with the hashes written when the cache was built."""
    info = json.loads((cache_dir / "cache_info.json").read_text(encoding="utf-8"))
    hashes = {}
    for name in names:
        actual = file_sha256(cache_dir / name)
        expected = info["files"][name]["sha256"]
        if actual != expected:
            raise SystemExit(f"{name} differs from the cache that was built and checked locally")
        hashes[name] = actual
    return {"normalizer_version": info["normalizer_version"], "sha256": hashes}


def write_outputs_manifest(output_dir: Path) -> Path:
    """List every output file with its hash, so a copy can be checked file by file."""
    files = sorted(
        p
        for p in output_dir.rglob("*")
        if p.is_file() and p.name != "outputs_manifest.json" and p.suffix != ".tmp"  # .tmp = unfinished
    )
    manifest = {
        "code": git_commit(),
        "environment": environment(),
        "files": {str(p.relative_to(output_dir)): file_sha256(p) for p in files},
    }
    path = output_dir / "outputs_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
