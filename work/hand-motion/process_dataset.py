"""Export only completed hand attempts and audit image/label correspondence."""
import argparse,json,subprocess,sys,time
from pathlib import Path
r=Path('outputs/hand-motion');start=time.monotonic();completed=0
for root in sorted(r.glob('hand-data-v1-seed*/seed*')):
 if not (root/'result.json').exists() or not (root/'physics-frames.npz').exists():continue
 if (root/'object-motion.json').exists() and (root/'image-render.json').exists():continue
 subprocess.run([sys.executable,str(Path(__file__).with_name('export_motion.py')),str(root)],check=True);completed+=1
print(json.dumps({'exported_scenes':completed,'seconds':time.monotonic()-start}))
