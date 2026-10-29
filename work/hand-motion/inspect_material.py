"""Measure material-wrapper side effects without starting physics."""
import json,numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.stage import open_stage_async,get_current_stage
from omni.kit.app import get_app
async def main():
 try:
  await open_stage_async('/opt/BrickSim/.runtime/hand-motion/hand-flat.usda');s=get_current_stage()
  for p in list(s.Traverse()):
   if p.IsInstance():p.SetInstanceable(False)
  pad=PhysicsMaterial('/diagnostic_material');rows=[]
  from pxr import UsdPhysics
  for p in list(s.Traverse()):
   if not p.HasAPI(UsdPhysics.CollisionAPI):continue
   before=np.array(UsdGeom.Xformable(p).GetLocalTransformation());SingleGeometryPrim(str(p.GetPath())).apply_physics_material(pad);after=np.array(UsdGeom.Xformable(p).GetLocalTransformation());rows.append({'path':str(p.GetPath()),'max_matrix_change':float(abs(before-after).max()),'before':before.tolist(),'after':after.tolist()})
  Path('/opt/BrickSim/.runtime/hand-motion/material-diagnostic.json').write_text(json.dumps(rows,indent=2));print('MATERIAL_DIAGNOSTIC',[(r['path'],r['max_matrix_change']) for r in rows],flush=True)
 except Exception:
  import traceback;Path('/opt/BrickSim/.runtime/hand-motion/material-diagnostic-error.txt').write_text(traceback.format_exc())
 get_app().post_quit(0)
