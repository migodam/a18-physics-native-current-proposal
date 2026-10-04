"""Anonymous fixed-commit and Release bytes; no credentials in HTTP requests."""
import argparse,concurrent.futures,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
ROOT=Path(__file__).resolve().parents[1]
REPO='migodam/a18-physics-native-current-proposal'
TAG='pilot-2026-10-04'
FILES=['START_HERE.md','RESEARCH_REPORT_ZH.md','RESULTS_SUMMARY.json','GATE_DECISION.json','results/frozen_evaluation/SUMMARY.json','results/frozen_evaluation/pilot_locked_test_00_state0_k8.json']
def command(args):return subprocess.check_output(args,cwd=ROOT)
def anonymous(url,expected):
    opener=build_opener(ProxyHandler({}))  # No environment proxy, netrc, or auth handler.
    h=hashlib.sha256();size=0
    with opener.open(Request(url,headers={'User-Agent':'A18-publication-verification/1.0'}),timeout=60) as r:
        status=r.status
        while True:
            chunk=r.read(1048576)
            if not chunk:break
            h.update(chunk);size+=len(chunk)
    actual=h.hexdigest()
    if actual!=expected:raise RuntimeError('Anonymous byte hash mismatch')
    return {'url':url,'HTTP_status':status,'bytes':size,'sha256':actual,'expected_sha256':expected,'status':'PASS','authentication':'NONE; environment proxies disabled; no netrc/auth handler'}
def run(args):
    commit=args.commit or command(['git','rev-parse','HEAD']).decode().strip()
    remote=command(['git','ls-remote','origin','refs/heads/main']).decode().split()[0]
    if remote!=commit:raise RuntimeError('Remote main differs from requested final commit')
    tasks=[]
    for name in FILES:
        expected=hashlib.sha256(command(['git','show',commit+':'+name])).hexdigest();tasks.append(('git:'+name,'https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+name,expected))
    assets=[]
    if not args.skip_release:
        release=json.loads(command(['gh','api','repos/'+REPO+'/releases/tags/'+TAG]));manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text())
        if release['draft']:raise RuntimeError('Release remains draft')
        for record in manifest['assets']:
            a=next(a for a in release['assets'] if a['name']==record['name'])
            if a['size']!=record['bytes']:raise RuntimeError('Release size mismatch')
            if a.get('digest') and a['digest']!='sha256:'+record['sha256']:raise RuntimeError('GitHub server digest mismatch')
            assets.append({'name':a['name'],'server_digest':a.get('digest'),'API_size':a['size']});tasks.append(('release:'+a['name'],a['browser_download_url'],record['sha256']))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        work={pool.submit(anonymous,url,expected):name for name,url,expected in tasks}
        for future in concurrent.futures.as_completed(work):rows.append({'artifact':work[future],**future.result()})
    result={'status':'PASS','repository':'https://github.com/'+REPO,'visibility':'PUBLIC_ANONYMOUS_READ_VERIFIED','commit':commit,'remote_main':remote,'release':'https://github.com/'+REPO+'/releases/tag/'+TAG if not args.skip_release else 'NOT_CHECKED_THIS_PASS','verified_utc':datetime.now(timezone.utc).isoformat(),'files':sorted(rows,key=lambda r:r['artifact']),'Release_API_assets':assets,'scientific_conclusion':'NO_GO_BOTH','entry':'https://raw.githubusercontent.com/'+REPO+'/'+commit+'/START_HERE.md','scope':'Publication/integrity verification, not scientific gate acceptance'}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','commit':commit,'files_verified':len(rows),'release_checked':not args.skip_release,'anonymous_bytes_verified':sum(r['bytes'] for r in rows)}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit');p.add_argument('--skip-release',action='store_true');p.add_argument('--output',required=True);run(p.parse_args())
