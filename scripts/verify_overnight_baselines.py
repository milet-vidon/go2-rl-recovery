"""Read-only byte-preservation guard, NOT behavioral acceptance."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'configs/overnight_preservation_20260917.json').read_text(encoding='utf-8'))
assert manifest['schema_version'] == 'overnight_preservation_v1'
for item in manifest['files']:
    path = Path(item['path'])
    assert path.is_file(), f'Missing preserved baseline: {path}'
    assert path.stat().st_size == item['bytes'], f'Baseline size changed: {path}'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == item['sha256'], f'Baseline hash changed: {path}'
print(f"All {len(manifest['files'])} preserved model files unchanged. This does not verify candidate capabilities.")
