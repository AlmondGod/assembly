import ast,os,signal,subprocess,time
from pathlib import Path
root=Path('/opt/BrickSim/.runtime/hand-motion');snapshot=root
node=next(n.value for n in ast.parse((root/'supervisor.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='command' for x in n.targets));command=eval(compile(ast.Expression(node),'supervisor-command','eval')).replace(str(root/'session.py:main'),str(root/'inspect_material.py:main'))
with (root/'material-diagnostic.log').open('w') as log:
 p=subprocess.Popen(['bash','-c',command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'));(root/'material-diagnostic-owned.json').write_text(__import__('json').dumps({'group':p.pid}))
 try:p.wait(timeout=90)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
 (root/'material-diagnostic-cleanup.json').write_text(__import__('json').dumps({'stopped':p.poll() is not None,'returncode':p.returncode}))
