"""Fit a shared wrist pose and pinch that keep the whole hand above the table."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinematics import HandKinematics
r=Path('outputs/hand-motion');k=HandKinematics(json.load(open(r/'hand-asset.json')));old=json.load(open(r/'pinch.json'));a=np.array(old['open']);b=np.array(old['closed']);active=[i for i,n in enumerate(k.names) if '_FF' in n or '_TH' in n];n=len(active)
for i in range(len(k.names)):
 if i not in active:a[i]=b[i]=1.3 if k.names[i][-1] in '012' else 0.
cloud=[(name,np.array(points)[::4]) for name,points in json.load(open(r/'hand-collision-cloud.json')).items()]
initial_r=Rotation.from_euler('x',105,degrees=True);h=.012;initial_p=np.array([0,0,h])-initial_r.apply(old['center']);seed=np.r_[a[active],b[active],initial_r.as_rotvec(),initial_p];low=np.r_[k.low[active]+1e-6,k.low[active]+1e-6,[-4,-4,-4],[-.3,-.3,.015]];high=np.r_[k.high[active]-1e-6,k.high[active]-1e-6,[4,4,4],[.3,.3,.4]];seed=np.clip(seed,low+1e-8,high-1e-8)
def residual(v):
 """Penalize fingertip target errors and any hand-box penetration of the table."""
 a[active]=v[:n];b[active]=v[n:2*n];rotation=Rotation.from_rotvec(v[-6:-3]);p=v[-3:];out=[]
 for t in np.linspace(0,1,5):
  q=a+t*(b-a);width=.046+t*(.022-.046)
  target=np.array([[-width/2,0,h],[width/2,0,h]]);out.extend(200*(rotation.apply(k.tips(q))+p-target).ravel()) if t in [0,1] else None;frames=k.forward(q)
  for name,points in cloud:
   local=points@frames[name][:3,:3].T+frames[name][:3,3];z=(rotation.apply(local)+p)[:,2];out.extend(200*np.minimum(z-(.015 if any(f in name for f in ['mf','rf','lf']) else .001),0))
 out.extend(.0001*(v[:2*n]-seed[:2*n]));out.extend(.001*(v[-6:]-seed[-6:]));return np.array(out)
fit=least_squares(residual,seed,bounds=(low,high),max_nfev=700,ftol=1e-9,xtol=1e-9,gtol=1e-9);loss=residual(fit.x);rot=Rotation.from_rotvec(fit.x[-6:-3]);p=fit.x[-3:];center=rot.inv().apply(np.array([0,0,h])-p);out={'names':k.names,'open':a.tolist(),'closed':b.tolist(),'center':center.tolist(),'wrist_rotation':rot.as_matrix().tolist(),'grasp_height':h,'palm_position_relative_to_brick':p.tolist(),'max_geometric_residual_m':float(max(abs(loss[:380]))/200),'cost':float(fit.cost)};(r/'pinch-collision.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
