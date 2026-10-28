import json,tarfile,hashlib,shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from research.repair_v8.prepare import sha,write_json
from research.repair_v8.qualify import TASKS,SEEDS,PROJECT,summarize_task,verify_files

remote_project=Path('/root/autodl-tmp/repair-v8-batch1-20261019T011025Z')
local=PROJECT/'runs/repair-v8-qualification-20261018/remote'
evidence=PROJECT/'evidence/repair-v8-gpu-qualification-batch1-20261019'
info=json.loads((evidence/'archive.json').read_text())
archive=local/'evidence.tgz'
assert sha(archive)==info['sha256'] and archive.stat().st_size==info['bytes']
collected=local/'collected';collected.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert len({m.name for m in members})==len(members)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    tar.extractall(collected,filter='data')
root=collected/'runs/repair-v8-qualification-20261018'
run=root/'gpu-controls/20261019T012023.952866Z'
rows=json.loads((run/'results.json').read_text())
plan=json.loads((root/'qualification-plan.json').read_text())
verify_files(PROJECT,plan['files']);verify_files(collected,plan['files'])
expected=set()
for key,task in TASKS.items():
    for seed in SEEDS:
        for revision in ['before','fixed']:
            checks=[('upstream','suite')]+[('probe',case) for case in task['cases']]
            if revision=='fixed':checks += [('profile',task['profile_case']),('memcheck',task['memcheck_case'])]
            for kind,case in checks:expected.add((key,seed,revision,kind,case))
observed=[tuple(r[k] for k in ['task','seed','revision','kind','case']) for r in rows]
assert len(observed)==len(set(observed)) and set(observed)==expected
assert len(rows)==204
for row in rows:
    log=collected/Path(row['process']['log']).relative_to(remote_project)
    assert sha(log)==row['process']['log_sha256'],str(log)
    sidecar=log.with_suffix('.json')
    if sidecar.exists():assert json.loads(sidecar.read_text())==row['receipt']
    else:assert row['receipt']['status']=='missing_receipt'
for key,task in TASKS.items():
    for revision in ['before','fixed']:
        manifest=json.loads((root/'execution-manifests'/(task['slug']+'-'+revision+'.json')).read_text())
        verify_files(PROJECT/'runs/repair-v8-qualification-20261018/execution'/task['slug']/revision,manifest['files'])
environment=json.loads((run/'environment.json').read_text())
computed=[summarize_task(key,[r for r in rows if r['task']==key],bool(environment['compute_sanitizer'])) for key in TASKS]
assert computed==json.loads((run/'summary.json').read_text())
report=json.loads((root/'controller/report.json').read_text())
assert report['status']=='complete' and report['error'] is None and report['tasks']==computed
for source,dest in [(run/'summary.json','task-summary.json'),(run/'environment.json','environment.json'),(root/'controller/report.json','report.json'),(root/'qualification-plan.json','qualification-plan.json')]:
    shutil.copy2(source,evidence/dest)
failures=Counter((r['task'],r['revision'],r['receipt'].get('status'),r['receipt'].get('error')) for r in rows if r['kind']=='probe' and r['process']['returncode']!=0)
write_json(evidence/'observed-probe-failures.json',[{'task':k[0],'revision':k[1],'status':k[2],'error':k[3],'count':v} for k,v in failures.items()])
upstream={key:dict(Counter(x['outcome'] for r in rows if r['task']==key and r['kind']=='upstream' for x in r['receipt'].get('rows',[]))) for key in TASKS}
audit={'utc':datetime.now(timezone.utc).isoformat(),'status':'pass','archive_sha256':info['sha256'],'archive_bytes':info['bytes'],'regular_members_verified':len(members),'planned_processes':len(expected),'actual_unique_processes':len(rows),'process_logs_hash_verified':len(rows),'all_receipt_sidecars_match':True,'frozen_runner_and_lock_match_local_and_remote':True,'execution_manifest_files_match_original_local_preparation':True,'task_summaries_independently_recomputed':True,'qualified_development_tasks':sum(x['status']=='development_controls_pass' for x in computed),'formal_eligible_tasks':0,'model_runs':0,'timeouts':sum(r['process']['timeout'] for r in rows),'original_test_outcome_counts':upstream}
write_json(evidence/'backup-audit.json',audit)
write_json(local/'backup-local-ack.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':info['sha256'],'local_archive':str(archive),'audit':str(evidence/'backup-audit.json')})
print(json.dumps(audit,indent=2))
