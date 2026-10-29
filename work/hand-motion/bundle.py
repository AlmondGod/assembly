"""Bundle the fresh source dataset, proof video, and exact collection code."""
import hashlib,json,zipfile
from pathlib import Path
r=Path('outputs/hand-motion');files=[]
for folder in [r/'dataset',r/'pilot024',*r.glob('hand-data-v1-seed*')]:
 files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ['.log'])
files.extend(r/n for n in ['report.json','hand-asset.json','hand-flat.usda','pinch-collision.json','hand-collision-cloud.json','material-diagnostic.json','shared-source-provenance.json','shared-demo.py','shared-basket.py','cleanup-audit.json','video-export-progress.json'])
manifest={str(p.relative_to(r)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files};(r/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(r/'bundle-manifest.json')
with zipfile.ZipFile(r/'hand-motion-pilot.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 for p in files:z.write(p,str(p.relative_to(r)))
print('BUNDLE',len(files),'files',(r/'hand-motion-pilot.zip').stat().st_size,'bytes')
