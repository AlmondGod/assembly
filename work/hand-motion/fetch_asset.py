"""Download USD layer dependencies and inspect the official articulated hand."""
import hashlib,json,urllib.request,urllib.parse
from pathlib import Path
from pxr import Sdf,Usd,UsdGeom,UsdPhysics
import numpy as np
ROOT=Path('/opt/BrickSim/.runtime/hand-motion');base='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/ShadowRobot/ShadowHand/';entry=base+'shadow_hand_instanceable.usd';queue=[entry];seen=set();manifest=[]
while queue:
 url=queue.pop(0)
 if url in seen:continue
 seen.add(url);p=ROOT/'assets'/urllib.parse.urlsplit(url).path.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
 layer=Sdf.Layer.FindOrOpen(str(p));refs=layer.GetExternalReferences();manifest.append({'url':url,'local':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});print('ASSET',p.name,p.stat().st_size,flush=True)
 for ref in refs:
  if ref.endswith(('.usd','.usda','.usdc')):queue.append(urllib.parse.urljoin(url,ref))
entry_path=ROOT/'assets'/urllib.parse.urlsplit(entry).path.lstrip('/');stage=Usd.Stage.Open(str(entry_path));cache=UsdGeom.XformCache();joints=[];bodies=[]
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):bodies.append({'path':str(prim.GetPath()),'pose':np.array(cache.GetLocalToWorldTransform(prim)).T.tolist()})
 if prim.IsA(UsdPhysics.Joint):
  j=UsdPhysics.Joint(prim);row={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'body0':[str(p) for p in j.GetBody0Rel().GetTargets()],'body1':[str(p) for p in j.GetBody1Rel().GetTargets()],'p0':list(j.GetLocalPos0Attr().Get()),'p1':list(j.GetLocalPos1Attr().Get()),'q0':[j.GetLocalRot0Attr().Get().GetReal(),*j.GetLocalRot0Attr().Get().GetImaginary()],'q1':[j.GetLocalRot1Attr().Get().GetReal(),*j.GetLocalRot1Attr().Get().GetImaginary()]}
  if prim.IsA(UsdPhysics.RevoluteJoint):
   rev=UsdPhysics.RevoluteJoint(prim);row.update(axis=rev.GetAxisAttr().Get(),lower=rev.GetLowerLimitAttr().Get(),upper=rev.GetUpperLimitAttr().Get())
  joints.append(row)
report={'entry':str(entry_path),'root':str(stage.GetDefaultPrim().GetPath()),'layers':manifest,'bodies':bodies,'joints':joints};(ROOT/'hand-asset.json').write_text(json.dumps(report,indent=2));stage.Flatten().Export(str(ROOT/'hand-flat.usda'));print('HAND_INSPECTED',len(bodies),len(joints),flush=True)
