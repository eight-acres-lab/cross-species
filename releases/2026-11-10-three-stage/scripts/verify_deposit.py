from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[1];m=json.loads((r/'DATA_MANIFEST.json').read_text())
for item in m['files']:
 p=r/item['path'];assert p.is_file(),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
print(f"Verified {len(m['files'])} code and data files")
