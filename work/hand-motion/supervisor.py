"""Run a finite isolated hand batch and stop only its owned process group."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path('/opt/BrickSim/.runtime/hand-motion');out=root/os.environ['HAND_RUN'];out.mkdir(exist_ok=False)
snapshot=out/'source';snapshot.mkdir()
for name in ['session.py','kinematics.py','recording.py','pinch.json','hand-boxes.json','hand-collision-cloud.json','hand-asset.json','hand-flat.usda','supervisor.py']:
 (snapshot/name).write_bytes((root/name).read_bytes())
(out/'source-hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},indent=2))
jobs=json.load(open(root/os.environ.get('HAND_JOBS','jobs.json')))
for i,j in enumerate(jobs):(out/f'job-{i:03}.json').write_text(json.dumps(j))
(out/f'job-{len(jobs):03}.json').write_text('{"finish":true}')
command=f'source /opt/bricksim-vulkan/env.sh; export OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH={snapshot}:/opt/BrickSim/python; exec /opt/BrickSim/.venv/bin/bricksim --exp isaacsim.exp.base.python --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 --/persistent/physics/numThreads=4 --/app/runLoops/main/rateLimitEnabled=false --/app/renderer/sleepMsOnFocus=0 --/app/renderer/sleepMsOutOfFocus=0 --/app/asyncRendering=false --/app/asyncRenderingLowLatency=false --/exts/isaacsim.core.throttling/enable_async=false --ext-folder /opt/BrickSim/python/bricksim/_exts --/app/window/enabled=false --/app/renderer/enabled=false --/renderer/enabled=false --/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true --/isaac/startup/ros_bridge_extension= {snapshot}/session.py:main'
started=time.monotonic()
with (out/'simulator.log').open('w') as log:
 p=subprocess.Popen(['bash','-c',command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,HAND_OUTPUT=str(out),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
 (out/'owned.json').write_text(json.dumps({'supervisor':os.getpid(),'group':p.pid}))
 try:p.wait(timeout=240+len(jobs)*300)
 except subprocess.TimeoutExpired:(out/'timeout.json').write_text('{"timeout":true}')
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  (out/'cleanup.json').write_text(json.dumps({'stopped':p.poll() is not None,'returncode':p.returncode,'seconds':time.monotonic()-started}))
