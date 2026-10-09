"""Verify the final publication manifest with standard-library Python only."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifest.json').read_text())
for item in manifest['files']:
    f=ROOT/item['path']
    assert f.is_file(),item['path']
    assert f.stat().st_size==item['bytes'],item['path']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['path']
print(f"Verified {len(manifest['files'])} published files.")
