"""List every output file with its hash; run at the end of each Colab session.

Writes outputs_manifest.json at the top of the output folder (DIGIKALA_OUTPUT_DIR), so
the copy on the local machine can be checked file by file with import_colab_outputs.py.

Usage:
    python scripts/write_outputs_manifest.py
"""

import json

from digikala_reviews.config import OUTPUT_ROOT, PROJECT_ROOT
from digikala_reviews.provenance import write_outputs_manifest


def main():
    if OUTPUT_ROOT == PROJECT_ROOT:
        raise SystemExit("Set DIGIKALA_OUTPUT_DIR to the output folder; this is meant for Colab runs")
    path = write_outputs_manifest(OUTPUT_ROOT)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"Listed {len(manifest['files'])} files from commit {manifest['code']['commit']} in {path}")


if __name__ == "__main__":
    main()
