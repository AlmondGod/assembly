"""Extract actual collision shapes in each hand body's coordinates."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
r=Path('/opt/BrickSim/.runtime/hand-motion');s=Usd.Stage.Open(str(r/'hand-flat.usda'));cache=UsdGeom.XformCache();out={}
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
 if not p.HasAPI(UsdPhysics.CollisionAPI):continue
 parent=p
 while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
 if not parent:continue
 name=parent.GetName()
 if name in ['robot0_hand_mount','robot0_forearm','robot0_wrist']:continue
 m=np.array(cache.ComputeRelativeTransform(p,parent)[0]).T;pts=[]
 if p.IsA(UsdGeom.Capsule):
  c=UsdGeom.Capsule(p);radius=c.GetRadiusAttr().Get();h=c.GetHeightAttr().Get()/2;axis='XYZ'.index(c.GetAxisAttr().Get());directions=np.array([[np.sin(t)*np.cos(a),np.sin(t)*np.sin(a),np.cos(t)] for t in np.linspace(0,np.pi,9) for a in np.linspace(0,2*np.pi,16,endpoint=False)])
  for sign in [-1,1]:
   x=directions*radius;x[:,axis]+=sign*h;pts.extend(x)
 elif p.IsA(UsdGeom.Cube):
  from itertools import product
  pts=np.array(list(product([-1,1],repeat=3)))*UsdGeom.Cube(p).GetSizeAttr().Get()/2
 else:
  print('UNHANDLED',p.GetPath(),p.GetTypeName());continue
 points=np.array(pts)@m[:3,:3].T+m[:3,3];out.setdefault(name,[]).extend(points.tolist())
(r/'hand-collision-cloud.json').write_text(json.dumps(out));print('COLLISION_CLOUD', {k:len(v) for k,v in out.items()})
