"""Bounded publication source/secret/asset audit; never print matched content."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATTERNS=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'sk-[A-Za-z0-9_-]{30,}',rb'AKIA[0-9A-Z]{16}']
def scan(blob):return any(re.search(p,blob) for p in PATTERNS)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def run():
    candidates=sorted(set(p for p in git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0') if p));issues=[];files=[]
    for rel in candidates:
        path=ROOT/rel
        if not path.is_file():continue
        b=path.read_bytes()
        if scan(b) or path.suffix.lower() in ('.pdf','.pem','.key') or any(s in rel for s in ('execution_private/','release_assets/','.ssh/')):issues.append({'scope':'candidate','path':rel})
        files.append({'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    # Scan all reachable branch history, without emitting old source contents.
    objects=git('rev-list','--objects','--all').splitlines();ids=[l.split(b' ',1)[0] for l in objects]
    metadata=subprocess.check_output(['git','cat-file','--batch-check'],input=b'\n'.join(ids)+b'\n',cwd=ROOT).decode().splitlines()
    blob_ids=[line.split()[0] for line in metadata if len(line.split())==3 and line.split()[1]=='blob'];data=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(blob_ids)+'\n').encode(),cwd=ROOT);at=0
    for oid in blob_ids:
        end=data.index(b'\n',at);header=data[at:end].decode().split();length=int(header[2]);blob=data[end+1:end+1+length];at=end+2+length
        if scan(blob):issues.append({'scope':'reachable_history','object':oid})
    archives=[];members=0
    for path in sorted((ROOT/'release_assets').glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                members+=1
                if scan(z.read(name)) or Path(name).suffix.lower() in ('.pdf','.pem','.key') or name.startswith('/') or '..' in Path(name).parts:issues.append({'scope':'release','archive':path.name,'member':name})
        archives.append({'name':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':'https://github.com/migodam/a18-physics-native-current-proposal/releases/download/pilot-2026-10-04/'+path.name})
    report={'status':'PASS' if not issues else 'BLOCKED','candidate_count':len(files),'reachable_history_blob_count':len(blob_ids),'release_archives':len(archives),'release_members_scanned':members,'issues':issues,'scope':'Known credential formats/private key markers and restricted extensions; manual source/rights review is RIGHTS_AND_PROVENANCE.md; not an exhaustive security proof','third_party_papers_redistributed':False,'license_change':False,'private_transport_profile_included':False,'full_source_original_modules':'Source hashes retained; A17 portable runtime path is disclosed'}
    (ROOT/'results/PUBLICATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={'schema':'a18.public_manifest.v1','research_date':'2026-10-04','scientific_conclusion':'NO_GO_BOTH','repository':'https://github.com/migodam/a18-physics-native-current-proposal','release_tag':'pilot-2026-10-04','assets':archives,'git_files':[f for f in files if f['path'] not in ('PUBLIC_MANIFEST.json','results/PUBLICATION_AUDIT.json','results/publication_receipt.json')],'self_hash_exclusion':'This manifest, audit and publication receipt excluded to avoid circular hashes; verify their bytes against the fixed Git commit','full_asset_members':'results/shared/REMOTE_RELEASE_MANIFEST.json; results/shared/A17_NUMERICAL_DEPENDENCY_MANIFEST.json','source_rights':'RIGHTS_AND_PROVENANCE.md','publication_verification':'results/publication_receipt.json and Release FINAL_PUBLICATION_QA.json'}
    (ROOT/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report));return not issues

if __name__=='__main__':raise SystemExit(0 if run() else 1)
