"""Back up only this experiment with content hashes on the persistent volume."""
import hashlib,json
from pathlib import Path
r=Path('/opt/BrickSim/.runtime/hand-motion');target=Path('/workspace/hand-motion');manifest={}
roots=[r/'hand-data-v1',r/'pilot024',*r.glob('hand-data-v1-seed*')]
files=[p for root in roots if root.is_dir() for p in root.rglob('*') if p.is_file()];files.extend(p for p in r.iterdir() if p.is_file() and (p.suffix in ['.py','.json','.usda']))
for p in files:
 relative=p.relative_to(r);data=p.read_bytes();out=target/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);digest=hashlib.sha256(data).hexdigest();assert hashlib.sha256(out.read_bytes()).hexdigest()==digest;manifest[str(relative)]={'sha256':digest,'bytes':len(data)}
(target/'backup-manifest.json').write_text(json.dumps(manifest,indent=2));print('HAND_BACKUP_OK',len(manifest),sum(x['bytes'] for x in manifest.values()))
