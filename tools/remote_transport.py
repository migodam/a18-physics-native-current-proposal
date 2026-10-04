"""Reuse an approved SSH stdin profile; credential-bearing argv is never logged.

Set A18_SSH_PROFILE_SOURCE privately. This module contains no credential, host,
user, or SSH-key path. All deployed numerical source is immutable by version.
"""
import ast,base64,hashlib,json,os,subprocess
from pathlib import Path

REMOTE_ROOT="D:/AI/A18_PHYSICS_NATIVE_CURRENT_PROPOSAL"

def call(script,timeout=60):
    p=Path(os.environ["A18_SSH_PROFILE_SOURCE"]);tree=ast.parse(p.read_text())
    cmd=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="cmd" for t in n.targets))
    if cmd[0]!="ssh" or cmd[-1]!="-" or "BatchMode=yes" not in cmd:raise RuntimeError("approved transport incompatible")
    r=subprocess.run(cmd,input=script.encode(),capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError("SSH transport failed (stderr withheld), code="+str(r.returncode))
    return r.stdout

def upload(files,version):
    payload=[]
    for rel,path in files.items():
        b=Path(path).read_bytes();payload.append({"path":version+"/"+rel,"sha256":hashlib.sha256(b).hexdigest(),"data":base64.b64encode(b).decode()})
    script="""import json,base64,hashlib\nfrom pathlib import Path\nr=Path(ROOT).resolve();rows=[]\nfor f in PAYLOAD:\n q=(r/f['path']).resolve()\n if not q.is_relative_to(r):raise RuntimeError('unsafe destination')\n b=base64.b64decode(f['data'])\n if hashlib.sha256(b).hexdigest()!=f['sha256']:raise RuntimeError('transport hash')\n if q.exists() and q.read_bytes()!=b:raise RuntimeError('immutable version replacement')\n q.parent.mkdir(parents=True,exist_ok=True)\n if not q.exists():q.write_bytes(b)\n rows.append({'path':f['path'],'sha256':f['sha256'],'bytes':len(b)})\nprint(json.dumps(rows))\n""".replace("ROOT",repr(REMOTE_ROOT)).replace("PAYLOAD",repr(payload))
    return json.loads(call(script,180))

def fetch(prefix,local_root):
    script="""import json,base64,hashlib\nfrom pathlib import Path\nr=Path(ROOT).resolve();s=(r/PREFIX).resolve()\nif not s.is_relative_to(r):raise RuntimeError('unsafe snapshot')\nfor q in sorted(s.rglob('*')):\n if q.is_file():\n  b=q.read_bytes();print(json.dumps({'path':q.relative_to(r).as_posix(),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()}))\n""".replace("ROOT",repr(REMOTE_ROOT)).replace("PREFIX",repr(prefix))
    rows=[]
    for line in call(script,180).decode().splitlines():
        a=json.loads(line);b=base64.b64decode(a.pop("data"));q=Path(local_root)/a["path"]
        if hashlib.sha256(b).hexdigest()!=a["sha256"]:raise RuntimeError("snapshot hash")
        q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);rows.append(a)
    return rows
