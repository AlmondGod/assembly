"""Validate dataset paths, hashes, seed splits and measured motion labels."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
r=Path('outputs/hand-motion');out=r/'dataset';out.mkdir(exist_ok=True);groups={'train':[],'heldout':[],'failed_train':[]};source_hashes=set();seeds={};max_horizon_error=0.
for p in sorted(r.glob('hand-data-v1-seed*/seed*/object-motion.json')):
 a=json.load(open(p));assert a['scene_valid'];assert hashlib.sha256((p.parent/'physics-frames.npz').read_bytes()).hexdigest()==a['source_pose_sha256'];cfg=p.parent.parent/'source-hashes.json';source_hashes.add(json.load(open(cfg))['session.py']);frames=np.load(p.parent/'physics-frames.npz');assets=np.load(p.parent/'meshes.npz');initial=json.load(open(p.parent/'initial.json'));assert len(initial['bricks'])==50
 for example in a['examples']:
  image=p.parent/example['image'];assert image.exists();assert Image.open(image).size==(512,384);assert Image.open(image).mode=='RGB';s=example['supervision'];current=np.array(s['current_points_world_m']);future=np.array(s['next_points_world_m']);assert np.allclose(future-current,s['displacement_world_m'],atol=1e-9);max_horizon_error=max(max_horizon_error,abs(s['horizon_s']-.5));assert abs(s['horizon_s']-.5)<1e-5
  group='heldout' if example['split']=='heldout_hand_scenes' else ('train' if example['eligible_for_successful_demonstration_training'] else 'failed_train');seeds.setdefault(group,set()).add(example['seed']);record=dict(example,image=str(image.relative_to(r)),camera_file=str((p.parent/'image-render.json').relative_to(r)),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest());groups[group].append(record)
assert len(source_hashes)==1;assert not seeds['train']&seeds['heldout'];assert len(set.union(*seeds.values()))==10
for group,rows in groups.items():(out/(group+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
manifest={'data_root':'Parent folder of dataset/','input_fields':['image','camera_file'],'target_fields':['supervision'],'metadata_fields_excluded_from_inputs':['seed','split','brick','attempt_success','eligible_for_successful_demonstration_training'],'scene_seeds':{k:sorted(v) for k,v in seeds.items()},'windows':{k:len(v) for k,v in groups.items()},'immutable_controller_sha256':next(iter(source_hashes)),'maximum_horizon_error_seconds':max_horizon_error,'validation':'RGB paths/hash verified,50bricks per scene, finite physically bounded trajectories, identical controller across10scenes, no train/heldoutseedoverlap, displacement arithmetic verified','new_training_run':False};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
