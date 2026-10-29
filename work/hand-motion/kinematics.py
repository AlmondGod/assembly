"""USD joint-frame forward kinematics and a two-fingertip pinch in metres."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class HandKinematics:
 def __init__(self,asset):
  """Load the inspected USD topology and retain the palm's finger subtree."""
  self.asset=asset;self.joints=[j for j in asset['joints'] if j['type']=='PhysicsRevoluteJoint' and '_WRJ' not in j['path']];self.names=[j['path'].split('/')[-1] for j in self.joints];self.low=np.deg2rad([j['lower'] for j in self.joints]);self.high=np.deg2rad([j['upper'] for j in self.joints]);self.frames=[]
  for j in self.joints:
   f=[]
   for side in ['0','1']:
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(j['q'+side],scalar_first=True).as_matrix();t[:3,3]=j['p'+side];f.append(t)
   self.frames.append(f)
 def forward(self,q):
  """Return finger body frames relative to the palm for radian joint angles."""
  poses={'robot0_palm':np.eye(4)}
  for j,frames,angle in zip(self.joints,self.frames,q):
   parent=j['body0'][0].split('/')[-1];child=j['body1'][0].split('/')[-1];axis=np.eye(3)['XYZ'.index(j['axis'])];m=np.eye(4);m[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix();poses[child]=poses[parent]@frames[0]@m@np.linalg.inv(frames[1])
  return poses
 def tips(self,q):
  """Return distal fingertip sample centers in the palm frame, shape2x3."""
  frames=self.forward(q);return np.array([(frames['robot0_'+s+'distal']@np.array([0,0,.012 if s=='ff' else .013,1]))[:3] for s in ['ff','th']])
 def solve(self,width,initial=None,center=(.035,-.025,.105)):
  """Fit index/thumb targets at a common fixed pinch center, with folded other fingers."""
  center=np.array(center);target=np.array([center+[-width/2,0,0],center+[width/2,0,0]]);q=np.zeros(len(self.names))
  for i,n in enumerate(self.names):
   if any('_'+s in n for s in ['MF','RF','LF']):q[i]=1.3 if n[-1] in '012' else 0.
  active=[i for i,n in enumerate(self.names) if '_FF' in n or '_TH' in n];seed=np.clip(np.array([.0,.5,.5,.3,.5,.8,.0,.0,-.8]),self.low[active]+1e-7,self.high[active]-1e-7) if initial is None else initial[active]
  def residual(v):
   q[active]=v;return np.r_[(self.tips(q)-target).ravel(),.0001*(v-seed)]
  result=least_squares(residual,seed,bounds=(self.low[active]+1e-8,self.high[active]-1e-8),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11);q[active]=result.x
  return q,{'width':width,'center':center.tolist(),'tip_positions':self.tips(q).tolist(),'target_positions':target.tolist(),'max_tip_error_m':float(np.max(np.linalg.norm(self.tips(q)-target,axis=1)))}

if __name__=='__main__':
 r=Path('outputs/hand-motion');k=HandKinematics(json.load(open(r/'hand-asset.json')));a,ar=k.solve(.046);b,br=k.solve(.022,a);(r/'pinch.json').write_text(json.dumps({'names':k.names,'open':a.tolist(),'closed':b.tolist(),'open_evidence':ar,'closed_evidence':br},indent=2));print(json.dumps({'open':ar,'closed':br},indent=2))
