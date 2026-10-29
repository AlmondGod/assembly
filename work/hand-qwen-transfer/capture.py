"""Capture matched YAM mesh/pose snapshots without generated policy execution."""
import inspect,json,os,runpy,textwrap,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.prims import RigidPrim
from omni.kit.app import get_app
from bricksim.cap.interactive import InteractiveSession
from bricksim.cap.yam import YamArm
from recording import PhysicsRecording
ROOT=Path('/opt/BrickSim/.runtime/hand-qwen-transfer');OUT=ROOT/'capture';OUT.mkdir(exist_ok=True)
os.environ.update(YAM_OUTPUT=str(OUT),YAM_BASKET='1',YAM_SUPPLY_COUNT='50',YAM_SEED='1024',YAM_CONTROL='interactive',YAM_BASKET_LAYOUT='compact')
Demo=runpy.run_path('/opt/BrickSim/demos/demo_yam_assembly.py')['Demo']
async def no_capture(self,*args):
 """Disable RTX capture for this isolated CPU-rendered experiment."""
 return None
Demo.capture=no_capture;Demo.run.__globals__['set_camera_view']=lambda *args:None
async def capture(self):
 """Save actual arm geometry, then old train and fresh settled test brick states."""
 from isaacsim.core.utils.stage import get_current_stage
 stage=get_current_stage();bricks=RigidPrim(list(self.supply_paths.values()),reset_xform_properties=False);bricks.initialize();order=[list(self.supply_paths.values()).index(p) for p in bricks.prim_paths];old=Path('/opt/BrickSim/.runtime/object-motion-scale/snapshots')
 for seed in list(range(1000,1012))+list(range(1024,1028)):
  out=OUT/f'seed{seed}';out.mkdir(exist_ok=False);self.output=out
  if seed<1024:state=json.load(open(old/f'seed{seed}.json'));poses={int(k):np.array(v) for k,v in state['bricks'].items()}
  else:poses=self.basket.scatter(seed)
  ids=list(self.supply_paths);positions=np.array([poses[ids[i]][:3,3] for i in order]);quats=np.array([Rotation.from_matrix(poses[ids[i]][:3,:3]).as_quat(scalar_first=True) for i in order]);bricks.set_world_poses(positions,quats);bricks.set_velocities(np.zeros((50,6)))
  if seed>=1024:
   for _ in range(240):await self.arms['left'].tick()
   state=self.snapshot(0)
  else:(out/'state-000.json').write_text(json.dumps(state,indent=2))
  rec=PhysicsRecording(stage,out,('/World/Yam_left','/World/Yam_right','/World/Basket',*self.supply_paths.values(),*self.paths.values()));rec.record(.05);rec.save();(out/'initial.json').write_text(json.dumps({'seed':seed,'brick_paths':self.supply_paths,'bricks':state['bricks']}));print('MATCHED_CAPTURE',seed,flush=True)
 get_app().post_quit(0)
InteractiveSession.run=capture
async def main():
 """Run the finite matched-image capture and save any failure."""
 try:await Demo().run()
 except Exception:(OUT/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();get_app().post_quit(1)
