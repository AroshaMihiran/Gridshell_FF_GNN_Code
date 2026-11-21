"""Verify SHA-256 checksums of the distributed source, inputs and model files."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/"release_manifest.json").read_text())
failures=[]
for item in manifest["files"]:
    path=root/item["file"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
        failures.append(item["file"])
if failures:
    raise SystemExit("Missing or changed files: "+", ".join(failures))
print(f"Verified {len(manifest['files'])} distributed files.")
