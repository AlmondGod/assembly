"""Render only measured hand/brick poses; no trajectory interpolation."""
import argparse,json,runpy,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from fast_simplification import simplify
from imageio_ffmpeg import get_ffmpeg_exe
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--preview',action='store_true');p.add_argument('--indices');p.add_argument('--width',type=int,default=960);args=p.parse_args();root=args.directory
from render_core import render_frame
assets=np.load(root/'meshes.npz');frames=np.load(root/'physics-frames.npz');meshes=[]
for i in range(int(assets['mesh_count'])):
 vertices=assets[f'v{i}'];faces=assets[f'f{i}']
 if len(faces)>1800:
  vertices,inverse=np.unique(np.round(vertices,7),axis=0,return_inverse=True);faces=inverse[faces].astype(np.int32);vertices,faces=simplify(vertices,faces,target_count=1800)
 body=int(assets[f'b{i}']);color=assets[f'c{i}'];name=str(assets['body_paths'][body]) if body>=0 else ''
 if '/managed/Parts/' in name:color=np.array([155,161,157])/255 if name.endswith('/Part_0') else np.array([201,26,9])/255
 elif body<0:color=np.array([.22,.32,.38])
 meshes.append((vertices,faces,body,color))
# Expand the native LEGO stud instancers using the repository's authoring dimensions.
# specs.cppm:8mm pitch,4.8mm diameter,1.7mm stud height; author.cppm adds50um visual tolerance.
for body,name in enumerate(assets['body_paths']):
 if '/managed/Parts/' not in str(name):continue
 nx,ny,top=(20,20,.00315) if str(name).endswith('/Part_0') else (2,4,.00955);v=[];f=[];segments=16
 for x in (np.arange(nx)-(nx-1)/2)*.008:
  for y in (np.arange(ny)-(ny-1)/2)*.008:
   base=len(v)
   for z in [top,top+.00175]:
    v.extend([[x+.0024*np.cos(t),y+.0024*np.sin(t),z] for t in np.arange(segments)*2*np.pi/segments])
   v.append([x,y,top+.00175]);cap=base+2*segments
   for j in range(segments):
    nxt=(j+1)%segments;f.extend([[base+j,base+nxt,base+segments+j],[base+nxt,base+segments+nxt,base+segments+j],[cap,base+segments+j,base+segments+nxt]])
 color=np.array([155,161,157])/255 if nx==20 else np.array([201,26,9])/255;meshes.append((np.array(v),np.array(f),body,color))
eye=np.array([.12,-.20,.345]);eye/=np.linalg.norm(eye);right=np.cross([0,0,1],eye);right/=np.linalg.norm(right);up=np.cross(eye,right);camera=np.array([right,up,eye]);center=np.array([.335,0,.105]);width=args.width;height=round(width*.75);start=time.monotonic();video=None;log=None
if not args.preview and not args.indices:
 log=(root/'ffmpeg.log').open('w');video=subprocess.Popen([get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'hand-physics.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
indices=[int(x) for x in args.indices.split(',')] if args.indices else ([0,len(frames['poses'])//2,len(frames['poses'])-1] if args.preview else range(len(frames['poses'])))
for n,index in enumerate(indices):
 triangles=[];colors=[]
 for vertices,faces,body,color in meshes:
  v=vertices
  if body>=0:
   pose=frames['poses'][index,body];v=vertices@pose[:3,:3].T+pose[:3,3]
  t=v[faces];normals=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10);light=np.array([.3,-.4,.85]);shade=.45+.55*np.abs(normals@light);triangles.append(t);colors.append(np.clip(shade[:,None]*color*255,0,255).astype(np.uint8))
 im=render_frame(np.concatenate(triangles),np.concatenate(colors),camera,center,1500.*width/960,width,height);draw=ImageDraw.Draw(im)
 if not args.indices:
  draw.text((15,12),f'Measured physics replay | Shadow Hand | t={frames["seconds"][index]:.2f}s',fill='white')
  result=json.load(open(root/'result.json'));draw.text((15,30),f'Seed {result["seed"]} | '+('Verified lift and hold' if result['success'] else 'Failed grasp attempt')+' | Scripted hand, dynamic bricks',fill='white')
 if args.preview or args.indices or index in [0,len(frames['poses'])//2,len(frames['poses'])-1]:im.save(root/f'{"frame" if args.indices else "video-frame"}-{index:04}.png')
 if video:video.stdin.write(im.tobytes())
 if n%50==0:print('RENDER',index,'/',len(frames['poses']),'wall',round(time.monotonic()-start,1),flush=True)
if video:
 video.stdin.close();rc=video.wait();log.close();assert rc==0
(root/('image-render.json' if args.indices else ('preview-render.json' if args.preview else 'video-render.json'))).write_text(json.dumps({'frames':len(frames['poses']),'sampled_physics_seconds':float(frames['seconds'][-1]-frames['seconds'][0]),'render_seconds':time.monotonic()-start,'input':'Measured live rigid-body transforms; no interpolation','renderer':'CPU orthographic, untextured meshes with native stud-instancer dimensions expanded at16segments','camera':{'basis':camera.tolist(),'center':center.tolist(),'pixels_per_metre':1500.*width/960,'width':width,'height':height}},indent=2))
