import subprocess,base64,io,tarfile,re,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/opt/BrickSim/.runtime/hand-motion'
def call(command):
 r=subprocess.run(['python3',str(ROOT/'work/pod_command.py')],input=command,text=True,capture_output=True,timeout=900)
 if r.returncode:raise RuntimeError(r.stderr[-1000:]+r.stdout[-1000:])
 return r.stdout

def upload(files):
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w:gz') as t:
  for path,name in files:t.add(path,arcname=name)
 encoded='\n'.join(textwrap.wrap(base64.b64encode(b.getvalue()).decode(),2048))
 result=call("python - <<'REMOTE'\nimport base64,io,tarfile,pathlib\nr=pathlib.Path("+repr(REMOTE)+");r.mkdir(parents=True,exist_ok=True)\npayload='''\n"+encoded+"\n'''\ntarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)),mode='r:gz').extractall(r,filter='data')\nprint('UPLOAD_OK')\nREMOTE\n")
 if 'UPLOAD_OK' not in result:raise RuntimeError(result[-2000:])

def fetch(patterns,destination):
 result=call("python - <<'REMOTE'\nimport glob,tarfile,io,base64,pathlib\nr=pathlib.Path("+repr(REMOTE)+");b=io.BytesIO()\nwith tarfile.open(fileobj=b,mode='w:gz') as t:\n for pat in "+repr(patterns)+":\n  for name in glob.glob(str(r/pat)):\n   p=pathlib.Path(name)\n   if p.is_file():t.add(p,arcname=str(p.relative_to(r)))\nprint('ARCHIVE:'+base64.b64encode(b.getvalue()).decode())\nREMOTE\n")
 m=re.search('ARCHIVE:([A-Za-z0-9+/=]+)',result)
 if not m:raise RuntimeError(result[-2000:])
 destination.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m[1])),mode='r:gz') as t:t.extractall(destination,filter='data')
