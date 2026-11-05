"""Drain active branches, validate their evidence and resume a frozen campaign."""
import argparse, os, sys, json, time, signal, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
parser=argparse.ArgumentParser(description='Drain current physical branches and resume with a runtime worker count')
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--parallel-cases',type=int,required=True)
args=parser.parse_args()
root=args.root.resolve()
workers=args.parallel_cases

launcher=root/'runtime/run_combined_parallel.py'
compile(launcher.read_text(),str(launcher),'exec')
plan=json.loads((root/'physical_plan.json').read_text())
assert 1<=workers<=len(plan['objects'])*len(plan['environment_seeds'])
for relative,expected in json.loads((root/'protocol_hashes.json').read_text()).items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
old_pid=int((root/'campaign.pid').read_text())
old_cmd=(Path('/proc')/str(old_pid)/'cmdline').read_bytes().split(b'\0')
assert any(str(root/name).encode() in old_cmd for name in ('code/combined_campaign.py','runtime/run_combined_parallel.py')),old_cmd
def running(pid):
    try:
        state=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]
        return state!='Z'
    except FileNotFoundError:
        return False
def physical_children():
    result=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv=folder.joinpath('cmdline').read_bytes().decode().split('\0')
            if str(root/'code/physical_simulation.py') in argv and running(int(folder.name)):
                result.append((int(folder.name),argv))
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            pass
    return result
stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
history={'started_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),'old_pid':old_pid,'new_parallel_cases':workers,'adopted_conditions':[]}
print('Stopping scheduler and draining its already-running simulation children',flush=True)
os.kill(old_pid,signal.SIGTERM)
deadline=time.monotonic()+30
while running(old_pid) and time.monotonic()<deadline:
    time.sleep(.2)
assert not running(old_pid),'Old scheduler did not exit'
for record_path in (root/'runtime').glob('execution_*.json'):
    previous=json.loads(record_path.read_text())
    if previous.get('status')=='running':
        previous.update(status='superseded',finished_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),next_parallel_cases=workers)
        record_path.write_text(json.dumps(previous,indent=2))
children=physical_children()
history['drained_child_pids']=[p for p,_ in children]
deadline=time.monotonic()+plan['child_timeout_seconds']+30
while any(running(pid) for pid,_ in children):
    if time.monotonic()>deadline:
        raise RuntimeError('An existing simulation child is still active; no replacement scheduler launched')
    time.sleep(.5)
progress=root/'campaign/campaign.json'
campaign=json.loads(progress.read_text())
(root/'runtime'/('campaign_before_parallel%d_'%workers+stamp+'.json')).write_text(json.dumps(campaign,indent=2))
plan_hash=hashlib.sha256((root/'physical_plan.json').read_bytes()).hexdigest()
for pid,argv in children:
    def argument(key,default=None):
        return argv[argv.index(key)+1] if key in argv else default
    target=Path(argument('--out')).resolve()
    assert root/'campaign' in target.parents
    report_path=target/'report.json'
    name=argument('--single-condition')
    archive=target/(name+'.npz')
    if not report_path.exists() or not archive.exists():
        continue
    report=json.loads(report_path.read_text())
    assert report['status']=='fresh_process_rollout_completed'
    assert report['source_plan_sha256']==plan_hash
    assert report['actor_parameters_unchanged'] and report['full_policy_state_unchanged']
    if argument('--reference'):
        assert report['prefix_gate_passed']
        assert report['reference_sha256']==hashlib.sha256(Path(argument('--reference')).read_bytes()).hexdigest()
    with np.load(archive,allow_pickle=False) as data:
        assert data['done'][-1] and not data['done'][:-1].any()
        assert all(np.isfinite(data[k]).all() for k in data.files)
        assert np.all(data['h_applied']>=0)
        np.testing.assert_array_equal(data['step'],np.arange(len(data['step'])))
    case=next(c for c in campaign['cases'] if c['object']==int(argument('--object')) and c['seed']==int(argument('--seed')))
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    matches=[r for r in case['runs'] if r['name']==name and r.get('npz_sha256')==archive_hash]
    if not matches:
        record={'name':name,'relative_dir':target.name,'exit_code':0}
        record.update(report)
        record.update(npz_sha256=archive_hash,report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
        case['runs'].append(record)
        history['adopted_conditions'].append({'object':case['object'],'seed':case['seed'],'name':name,'npz_sha256':archive_hash})
pending=progress.with_suffix('.runtime_switch.pending.json')
pending.write_text(json.dumps(campaign,indent=2))
pending.replace(progress)
history['completed_before_resume']=sum(len(c['runs']) for c in campaign['cases'])
env=os.environ.copy()
env.update(VK_ICD_FILENAMES='/hy-tmp/embodied_sae/userspace/nvidia_icd.json',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/hy-tmp/embodied_sae/.cache/tmp')
log_path=root/'runtime'/('campaign_parallel%d_'%workers+stamp+'.log')
with log_path.open('a') as log:
    process=subprocess.Popen(['/hy-tmp/embodied_sae/.venv/bin/python','-B','-u',str(launcher),'--root',str(root),'--parallel-cases',str(workers)],cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'campaign.pid').write_text(str(process.pid))
history.update(new_pid=process.pid,log_path=str(log_path),launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),completed_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat())
(root/'runtime'/('parallel_switch_'+stamp+'.json')).write_text(json.dumps(history,indent=2))
launch={'pid':process.pid,'started_beijing':history['completed_beijing'],'root':str(root),'parallel_cases':workers,'maximum_planned_rollouts':plan['maximum_planned_rollouts'],'previous_pid':old_pid,'log_path':str(log_path)}
(root/'launch.json').write_text(json.dumps(launch,indent=2))
print(json.dumps(history),flush=True)

