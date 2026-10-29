"""Physical Shadow Hand pinch trials in the existing compact brick basket."""
import asyncio,inspect,json,os,runpy,textwrap,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf,Sdf,PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdShade
from isaacsim.core.prims import RigidPrim,SingleArticulation,SingleGeometryPrim
from isaacsim.core.utils.types import ArticulationAction
from omni.kit.app import get_app
from bricksim.utils.physics_step import wait_for_physics_step
from recording import PhysicsRecording
from kinematics import HandKinematics
import omni.physics.tensors as physics_tensors
ROOT_HAND=Path(__file__).resolve().parent;OUT=Path(os.environ['HAND_OUTPUT']);OUT.mkdir(exist_ok=True)
os.environ.update(YAM_OUTPUT=str(OUT),YAM_BASKET='1',YAM_SUPPLY_COUNT='50',YAM_SEED=str(json.load(open(OUT/'job-000.json'))['seed']),YAM_CONTROL='interactive',YAM_BASKET_LAYOUT='compact')
namespace=runpy.run_path('/opt/BrickSim/demos/demo_yam_assembly.py');Demo=namespace['Demo'];asset=json.load(open(ROOT_HAND/'hand-asset.json'));pinch=json.load(open(ROOT_HAND/'pinch.json'))

async def hand_run(self,world,hand,stage):
 """Execute immutable file jobs with force-controlled wrist and driven fingers."""
 kinematics=HandKinematics(asset);body_paths=['/World/Hand/'+name for name in kinematics.forward(np.zeros(len(pinch['names'])))];simulation=physics_tensors.create_simulation_view('numpy');simulation.set_subspace_roots('/');hand_view=simulation.create_rigid_body_view(body_paths);indices=np.arange(len(body_paths),dtype=np.int32)
 palm=RigidPrim('/World/Hand/robot0_palm',reset_xform_properties=False);palm.initialize();tip_paths=['/World/Hand/robot0_'+f+'distal' for f in ['ff','th']]
 tips=RigidPrim(tip_paths,reset_xform_properties=False,track_contact_forces=True,contact_filter_prim_paths_expr=[list(self.supply_paths.values()) for _ in tip_paths]);tips.initialize();bricks=RigidPrim(list(self.supply_paths.values()),reset_xform_properties=False);bricks.initialize()
 assert list(bricks.prim_paths)==list(self.supply_paths.values()), (bricks.prim_paths,list(self.supply_paths.values()))
 assert list(hand_view.prim_paths)==body_paths
 palm_index=np.array([body_paths.index('/World/Hand/robot0_palm')],dtype=np.int32)
 drives={n:UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Hand/joints/'+n),'angular') for n in pinch['names']}
 names=pinch['names'];open_q=np.array(pinch['open']);closed_q=np.array(pinch['closed'])
 self.desired_q=open_q.copy();self.desired_p=np.array([.33,0,.35]);self.desired_r=Rotation.from_matrix(np.diag([1.,-1.,-1.]));self.desired_v=np.zeros(3);self.recording=None;self.grasp_active=False;self.pinch_alpha=0.;self.pinch_index=None;self.tick_number=0;dt=world.get_physics_dt();start=time.monotonic()
 def transforms():
  local=kinematics.forward(self.desired_q);base=np.eye(4);base[:3,:3]=self.desired_r.as_matrix();base[:3,3]=self.desired_p;matrices=np.array([base@local[path.split('/')[-1]] for path in body_paths]);return np.column_stack([matrices[:,:3,3],Rotation.from_matrix(matrices[:,:3,:3]).as_quat()]).astype(np.float32)
 def measured_angles():
  data=hand_view.get_transforms();matrices={path.split('/')[-1]:np.block([[Rotation.from_quat(row[3:]).as_matrix(),row[:3,None]],[np.zeros((1,3)),np.ones((1,1))]]) for path,row in zip(body_paths,data)};angles=[]
  for j,frames in zip(kinematics.joints,kinematics.frames):
   parent=j['body0'][0].split('/')[-1];child=j['body1'][0].split('/')[-1];rel=np.linalg.inv(frames[0])@np.linalg.inv(matrices[parent])@matrices[child]@frames[1];angles.append(float(Rotation.from_matrix(rel[:3,:3]).as_rotvec()['XYZ'.index(j['axis'])]))
  return angles
 def record_step(step_dt):
  if self.recording:self.recording.record(step_dt)
 world.add_physics_callback('hand_recording',record_step)
 async def tick():
  if self.grasp_active:
   forces=np.linalg.norm(tips.get_contact_force_matrix(dt=dt)[:,self.pinch_index,:],axis=1)
   if min(forces)<.03:self.pinch_alpha=min(1.,self.pinch_alpha+.002)
   self.desired_q=open_q+self.pinch_alpha*(np.array(pinch['closed'])-open_q)
  hand_view.set_kinematic_targets(transforms(),indices)
  await wait_for_physics_step(world);self.tick_number+=1
  if self.tick_number%120==0:print('HAND_TICK',self.tick_number,'wall',time.monotonic()-start,'palm',palm.get_world_poses()[0].tolist(),flush=True)
 async def segment(label,target_p,target_q,steps):
  """Interpolate wrist and finger drive targets while preserving dynamic contacts."""
  initial_p=self.desired_p.copy();initial_q=self.desired_q.copy();prior=initial_p.copy();print('HAND_ACTION',label,flush=True)
  for i in range(steps):
   t=(i+1)/steps;s=t*t*t*(10-15*t+6*t*t);self.desired_p=initial_p+s*(target_p-initial_p);self.desired_v=(self.desired_p-prior)/dt;prior=self.desired_p.copy();self.desired_q=initial_q+s*(target_q-initial_q) if not self.grasp_active else self.desired_q;await tick()
   if label=='close':
    force=np.linalg.norm(tips.get_contact_force_matrix(dt=dt),axis=2);bilateral=np.where(np.min(force,axis=0)>.005)[0]
    if len(bilateral):
     s2=min(1,s+.008);self.desired_q=initial_q+s2*(target_q-initial_q);closed_q[:]=self.desired_q;self.pinch_alpha=s2;self.pinch_index=int(bilateral[0]);self.grasp_active=True;await tick();print('HAND_PINCH_CONTACT',bilateral.tolist(),force[:,bilateral].tolist(),flush=True);break
  self.desired_v=np.zeros(3)
  for _ in range(30):await tick()
  p,q=palm.get_world_poses();actual=np.array([p[0],self.desired_p]);print('HAND_ENDPOINT',label,actual.tolist(),flush=True)
  return {'label':label,'palm_position':p[0].tolist(),'target_position':self.desired_p.tolist(),'prescribed_joint_angles':self.desired_q.tolist(),'measured_joint_angles':measured_angles(),'tip_positions':tips.get_world_poses()[0].tolist(),'contact_force_matrix':tips.get_contact_force_matrix(dt=dt).tolist()}
 (OUT/'ready.json').write_text(json.dumps({'dofs':names,'root':'/World/Hand/robot0_palm','pinch':pinch}))
 for job_index in range(30):
  deadline=time.monotonic()+900
  while not (OUT/f'job-{job_index:03}.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('Hand collector idle deadline')
   await asyncio.sleep(.2)
  job=json.load(open(OUT/f'job-{job_index:03}.json'))
  if job.get('finish'):return
  run=OUT/job['id'];run.mkdir(exist_ok=False);self.recording=None;self.grasp_active=False;closed_q[:]=pinch['closed']
  if job_index:raise RuntimeError('Fresh simulator required per basket seed until reset stability is validated')
  self.desired_p=np.array([.33,0,.35]);self.desired_r=Rotation.from_matrix(np.diag([1.,-1.,-1.]));self.desired_q=open_q.copy();self.desired_v=np.zeros(3)
  print('HAND_INITIAL','palm',palm.get_world_poses()[0].tolist(),'dt',dt,flush=True)
  for _ in range(round(4/dt)):await tick()
  positions,orientations=bricks.get_world_poses();before={str(k):np.block([[Rotation.from_quat(q,scalar_first=True).as_matrix(),p[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist() for k,p,q in zip(self.supply_paths,positions,orientations)}
  (run/'settled.json').write_text(json.dumps({'bricks':before,'palm':palm.get_world_poses()[0].tolist(),'controller':'kinematic hand with compliant contacts; dynamic bricks'},indent=2))
  print('HAND_SETTLED',positions.min(axis=0).tolist(),positions.max(axis=0).tolist(),'upright',sum(np.array(m)[2,2]>.88 for m in before.values()),flush=True)
  candidates=[k for k,m in before.items() if m[2][2]>.88];candidates.sort(key=lambda k:-before[k][2][3]);center=np.array(pinch['center']);boxes=json.load(open(ROOT_HAND/'hand-collision-cloud.json'));plans=[]
  for key in candidates:
   brick=np.array(before[key]);yaw=np.arctan2(brick[1,0],brick[0,0])
   for flip in [0,np.pi]:
    r=Rotation.from_euler('z',yaw+flip).as_matrix()@np.array(pinch['wrist_rotation']);grasp=brick[:3,3]+brick[:3,2]*job.get('height',pinch['grasp_height']);base=grasp-r@center;cloud=[]
    for t in np.linspace(0,1,7):
     local=kinematics.forward(open_q+t*(closed_q-open_q))
     for name,points in boxes.items():
      m=local[name];cloud.extend((np.array(points)@m[:3,:3].T+m[:3,3])@r.T+base)
    cloud=np.array(cloud);low=cloud[cloud[:,2]<.078];clearance=min(float(cloud[:,2].min()-.003),float((low[:,0]-.247).min()),float((.423-low[:,0]).min()),float((low[:,1]+.118).min()),float((.118-low[:,1]).min())) if len(low) else .1
    plans.append({'brick':key,'rotation':r,'base':base,'clearance':clearance,'height':brick[2,3]})
  valid=[x for x in plans if x['clearance']>=-.0003];valid.sort(key=lambda x:(-x['height'],-x['clearance']))
  (run/'planning.json').write_text(json.dumps([dict(x,rotation=x['rotation'].tolist(),base=x['base'].tolist()) for x in plans],indent=2))
  if not valid:raise RuntimeError('No wall/floor-clear hand approach; best clearance='+str(max(x['clearance'] for x in plans)))
  plan=valid[job.get('rank',0)];selected=plan['brick'];r=plan['rotation'];base=plan['base'];self.desired_r=Rotation.from_matrix(r);hover=base+np.array([0,0,.13]);self.desired_p=hover.copy()
  for _ in range(90):await tick()
  self.recording=PhysicsRecording(stage,run,('/World/Hand','/World/Basket',*self.supply_paths.values(),*self.paths.values()));actions=[]
  (run/'initial.json').write_text(json.dumps({'seed':job['seed'],'selected':selected,'brick_paths':self.supply_paths,'clearance_m':plan['clearance'],'bricks':before,'pinch':pinch,'palm_grasp':base.tolist(),'palm_rotation':r.tolist()},indent=2))
  actions.append(await segment('descend',base,open_q,round(4/dt)));actions.append(await segment('close',base,closed_q,round(2/dt)));actions.append(await segment('lift',hover,closed_q,round(4/dt)));lift_positions=bricks.get_world_poses()[0].copy()
  for _ in range(round(1/dt)):await tick()
  final_positions,final_q=bricks.get_world_poses();contact=tips.get_contact_force_matrix(dt=dt);rows=[]
  for i,(k,p,q) in enumerate(zip(self.supply_paths,final_positions,final_q)):
   lift=float(p[2]-before[str(k)][2][3]);drift=float(np.linalg.norm(p-lift_positions[i]));forces=np.linalg.norm(contact[:,i,:],axis=1)
   if lift>.04:rows.append({'brick':k,'lift_m':lift,'hold_drift_m':drift,'tip_contact_force_N':forces.tolist(),'position':p.tolist()})
  success=any(e['lift_m']>=.065 and e['hold_drift_m']<.003 and min(e['tip_contact_force_N'])>.001 for e in rows)
  self.recording.save();self.recording=None;result={'id':job['id'],'seed':job['seed'],'selected':selected,'success':success,'lift_evidence':rows,'actions':actions,'controller':'Prescribed22DOFkinematicShadowHand with compliant100N/mcontact material and contact feedback; dynamic bricks; no attachments or brick pose commands during actions','finger_contacts':'Normal contact-force matrix per index/thumb and brick;one simulated second holding','physics_dt_s':dt,'hold_ticks':round(1/dt),'spawn_layer_spacing_m':.013,'automatic_stud_snapping':False,'seconds':time.monotonic()-start};(run/'result.json').write_text(json.dumps(result,indent=2));(OUT/f'done-{job_index:03}.json').write_text(json.dumps(result,indent=2));print('HAND_RESULT',json.dumps(result),flush=True)

async def no_capture(self,*args):
 """Keep the pilot independent of RTX; measured poses are recorded for replay."""
 return None
Demo.capture=no_capture
source=textwrap.dedent(inspect.getsource(Demo.run));start=source.index('    urdf =');end=source.index('    thresholds =')
source=source[:start]+'''
    add_reference_to_stage(str(ROOT_HAND/'hand-flat.usda'),'/World/Hand')
    soft=PhysxSchema.PhysxMaterialAPI.Apply(pad.prim);soft.CreateCompliantContactStiffnessAttr(100.);soft.CreateCompliantContactDampingAttr(.02);soft.CreateCompliantContactAccelerationSpringAttr(False)
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith('/World/Hand') and prim.IsInstance():prim.SetInstanceable(False)
    removed=['robot0_hand_mount','robot0_forearm','robot0_wrist']
    for name in removed:stage.GetPrimAtPath('/World/Hand/'+name).SetActive(False)
    for prim in list(stage.Traverse()):
        if not str(prim.GetPath()).startswith('/World/Hand'):continue
        if prim.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(prim);targets=list(j.GetBody0Rel().GetTargets())+list(j.GetBody1Rel().GetTargets())
            if not j.GetBody0Rel().GetTargets() or any(p.name in removed for p in targets):prim.SetActive(False);continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                d=UsdPhysics.DriveAPI.Apply(prim,'angular');d.CreateTypeAttr('force');d.CreateStiffnessAttr(.03);d.CreateDampingAttr(.0015);d.CreateMaxForceAttr(.08);PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(180.)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateDisableGravityAttr(True);rb.CreateRetainAccelerationsAttr(False);rb.CreateMaxDepenetrationVelocityAttr(.5);rb.CreateMaxLinearVelocityAttr(2.);rb.CreateMaxAngularVelocityAttr(15.)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(json.load(open(OUT/'job-000.json')).get('hand_contacts',True))
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.0003);UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(pad.prim),UsdShade.Tokens.weakerThanDescendants,'physics')
    for prim in list(stage.Traverse()):
        if not str(prim.GetPath()).startswith('/World/Hand'):continue
        if prim.IsA(UsdPhysics.Joint):prim.SetActive(False)
        schemas=[a for a in prim.GetAppliedSchemas() if 'Tendon' not in a and 'Articulation' not in a]
        prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(schemas))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateSolverPositionIterationCountAttr(32);rb.CreateSolverVelocityIterationCountAttr(4)
    bodies=[p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Hand') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for p in bodies:UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets([b.GetPath() for b in bodies if b!=p])
    stage.GetPrimAtPath('/World/Hand').RemoveAPI(UsdPhysics.ArticulationRootAPI)
    fk=HandKinematics(asset).forward(np.array(pinch['open']));base=np.eye(4);base[:3,:3]=np.diag([1.,-1.,-1.]);base[:3,3]=[.33,0,.35]
    for name,local in fk.items():UsdGeom.Xformable(stage.GetPrimAtPath('/World/Hand/'+name)).MakeMatrixXform().Set(Gf.Matrix4d((base@local).T.tolist()))
    cache=UsdGeom.XformCache();shapes=[]
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith('/World/Hand') or not p.HasAPI(UsdPhysics.CollisionAPI):continue
        parent=p
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
        shapes.append(dict(path=str(p.GetPath()),body=str(parent.GetPath()),relative=np.array(cache.ComputeRelativeTransform(p,parent)[0]).T.tolist(),type=p.GetTypeName(),world=np.array(cache.GetLocalToWorldTransform(p)).T.tolist()))
    (OUT/'collision-setup.json').write_text(json.dumps(shapes,indent=2))
    hand=None
    scene=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene'));scene.CreateEnableGPUDynamicsAttr(False);scene.CreateBroadphaseTypeAttr('MBP')
'''+source[end:]
# Disable automatic stud snapping during pick-only source collection.
source=source.replace('thresholds.required_force = 1.0','thresholds.required_force = 1e9')
source=source.replace('    scattered.update(self.source_poses)', '''    for m in scattered.values():m[2,3]=self.basket.floor+.001+round((m[2,3]-self.basket.floor-.001)/.025)*.013
    scattered.update(self.source_poses)''')
# Preserve the visible floor surface while extending its collision thickness downward.
source=source.replace('("floor", [*center, basket.floor / 2], [*size, basket.floor])', '("floor", [*center, basket.floor - .025], [*size, .05])')
# Retain the exact basket/brick construction, then replace arm initialization and task.
start=source.index('    for name, side in [("left", -1), ("right", 1)]:',source.index('    await world.reset_async()'))
source=source[:start]+"    await hand_run(self,world,hand,stage)\n"
source=source.replace('    await world.reset_async()', '''    world.set_simulation_dt(physics_dt=1/120,rendering_dt=1/120)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene')).CreateEnableCCDAttr(True)
    for path in self.supply_paths.values():
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path));rb.CreateEnableCCDAttr(True);rb.CreateMaxDepenetrationVelocityAttr(.1);rb.CreateSolverPositionIterationCountAttr(8);rb.CreateSolverVelocityIterationCountAttr(2)
        for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0003);c.CreateRestOffsetAttr(0.)
    await world.reset_async()''')
# pad material is normally defined inside the removed arm block.
source=source.replace("    add_reference_to_stage(str(ROOT_HAND", "    pad=PhysicsMaterial('/World/HandPad',static_friction=2.5,dynamic_friction=2.,restitution=0)\n    add_reference_to_stage(str(ROOT_HAND")
namespace.update(globals());exec(compile(source,__file__,'exec'),namespace);Demo.run=namespace['run']
async def main():
 """Own the finite hand demonstration and expose failures to the supervisor."""
 try:await Demo().run()
 except Exception:
  (OUT/'failure.json').write_text(traceback.format_exc());traceback.print_exc();get_app().post_quit(1);return
 get_app().post_quit(0)
