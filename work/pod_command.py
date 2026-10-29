"""Execute a supplied command through RunPod's PTY-only SSH gateway."""
import os,select,subprocess,sys,time
p=subprocess.Popen(['ssh','-tt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15','-i',os.path.expanduser('~/.ssh/id_ed25519'),os.environ.get('BRICKSIM_SSH_TARGET','ozzzq5hjse3zov-64411314@ssh.runpod.io')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
b=bytearray(); end=time.monotonic()+45
while b':/# ' not in b:
    if time.monotonic()>end: p.kill(); raise TimeoutError('SSH login')
    if select.select([p.stdout],[],[],1)[0]:
        c=os.read(p.stdout.fileno(),8192)
        if not c: raise RuntimeError(b.decode(errors='replace'))
        b.extend(c)
command=sys.stdin.read()
out,_=p.communicate(('stty -echo\ncd /opt/BrickSim\n'+command+'\nexit\n').encode(),timeout=900)
print(out.decode(errors='replace').replace('\r',''))
