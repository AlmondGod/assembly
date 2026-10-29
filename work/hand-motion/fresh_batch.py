"""Collect independent seeded scenes with finite owned simulator processes."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/opt/BrickSim/.runtime/hand-motion');run='hand-data-v1';out=root/run;out.mkdir(exist_ok=False);start=time.monotonic();rows=[]
for seed in range(30010,30020):
 name=f'{run}-seed{seed}';jobs=root/f'{name}-jobs.json';jobs.write_text(json.dumps([{'id':f'seed{seed}','seed':seed,'hand_contacts':True}]))
 p=subprocess.Popen(['python',str(root/'supervisor.py')],env=dict(os.environ,HAND_RUN=name,HAND_JOBS=jobs.name));p.wait()
 folder=root/name;result=folder/'done-000.json';failure=folder/'failure.json';row={'seed':seed,'run':name,'elapsed_s':time.monotonic()-start,'returncode':p.returncode,'split':'heldout_hand_scenes' if seed>=30018 else 'source_train'}
 if result.exists():
  data=json.loads(result.read_text());row.update(success=data['success'],selected=data['selected'],lift_evidence=data['lift_evidence'])
 else:row.update(success=False,failure=failure.read_text() if failure.exists() else 'No result; inspect simulator log')
 rows.append(row);(out/'progress.json').write_text(json.dumps({'completed':len(rows),'successful':sum(x['success'] for x in rows),'wall_seconds':time.monotonic()-start,'rows':rows},indent=2));print('HAND_BATCH',seed,row['success'],flush=True)
(out/'finished.json').write_text(json.dumps({'finished':True,'wall_seconds':time.monotonic()-start}))
