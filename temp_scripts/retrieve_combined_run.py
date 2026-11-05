"""Stream the completed server run locally and verify every file before shutdown."""
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, 'D:/homework/DeepLearning/tmp/sshdeps')
import paramiko


def verify_local(local):
    archive = local / 'remote.tar.gz'
    destination = Path('\\\\?\\' + str(local / 'remote'))
    destination.mkdir(exist_ok=True)
    print('EXTRACTING WITH WINDOWS LONG PATH SUPPORT', flush=True)
    with tarfile.open(archive) as bundle:
        def normalized_members():
            for member in bundle:
                member.name = str(Path(member.name.removeprefix('./')))
                if member.name != '.':
                    yield member
        bundle.extractall(destination, members=normalized_members(), filter='data')
    manifest = json.loads((destination / 'runtime/retrieval_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = destination / name
        if path.stat().st_size != expected['size']:
            raise RuntimeError('Size differs: ' + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise RuntimeError('SHA256 differs: ' + name)
    verification = {'status': 'verified', 'files': len(manifest['files']),
                    'bytes': manifest['total_bytes'], 'archive_bytes': archive.stat().st_size,
                    'verified_unix': time.time(),
                    'remote_root': '/hy-tmp/embodied_sae/runs/combined_opening_20261103_140606_955714'}
    (local / 'retrieval_verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification), flush=True)


def main():
    project = Path(__file__).resolve().parents[1]
    local = project / 'outputs/combined_physical/20261103_140606_955714'
    if '--verify-local' in sys.argv:
        verify_local(local)
        return
    remote = '/hy-tmp/embodied_sae/runs/combined_opening_20261103_140606_955714'
    client = paramiko.SSHClient()
    client.load_host_keys(str(project / 'outputs/server_checks/20261102_152622/known_hosts'))
    password = getpass.getpass('SSH password: ')
    client.connect('i-2.gpushare.com', port=37840, username='root',
                   password=password, look_for_keys=False,
                   allow_agent=False, timeout=25)
    client.get_transport().default_window_size = 64 * 1024 * 1024
    try:
        source = '''import pathlib,json,hashlib,datetime
r=pathlib.Path(REMOTE)
s=json.loads((r/'campaign/campaign.json').read_text())
assert s['status']=='completed'
assert len(s['cases'])==8
assert all(c['status']=='completed' and len(c['runs'])==902 and all(x['exit_code']==0 for x in c['runs']) for c in s['cases'])
files={}
for p in sorted(r.rglob('*')):
 if p.is_symlink(): raise RuntimeError('Unexpected symlink '+str(p))
 if p.is_file() and p!=r/'runtime/retrieval_manifest.json':
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''): h.update(block)
  files[p.relative_to(r).as_posix()]={'size':p.stat().st_size,'sha256':h.hexdigest()}
m={'created_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':files,'total_bytes':sum(v['size'] for v in files.values())}
(r/'runtime/retrieval_manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'files':len(files),'bytes':m['total_bytes']}))
'''.replace('REMOTE', repr(remote))
        _, stdout, stderr = client.exec_command('/hy-tmp/embodied_sae/.venv/bin/python -c ' + shlex.quote(source), timeout=600)
        result = stdout.read().decode()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error)
        print('REMOTE MANIFEST ' + result, flush=True)
        archive = local / 'remote.tar.gz'
        partial = local / 'remote.tar.gz.partial'
        remote_archive = remote + '_retrieval.tar.gz'
        _, stdout, stderr = client.exec_command('tar -czf ' + shlex.quote(remote_archive) + ' -C ' + shlex.quote(remote) + ' .', timeout=600)
        stdout.read()
        error = stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error)
        with client.open_sftp() as sftp:
            size = sftp.stat(remote_archive).st_size
        print('ARCHIVE BYTES ' + str(size), flush=True)
        with partial.open('wb') as target:
            target.truncate(size)

        def download_part(index):
            connection = paramiko.SSHClient()
            connection.load_host_keys(str(project / 'outputs/server_checks/20261102_152622/known_hosts'))
            connection.connect('i-2.gpushare.com', port=37840, username='root', password=password,
                               look_for_keys=False, allow_agent=False, timeout=25)
            connection.get_transport().default_window_size = 64 * 1024 * 1024
            start = size * index // 8
            end = size * (index + 1) // 8
            try:
                with connection.open_sftp() as sftp, sftp.open(remote_archive, 'rb') as source, partial.open('r+b') as target:
                    source.seek(start)
                    source.prefetch(end, max_concurrent_requests=64)
                    target.seek(start)
                    position = start
                    last = time.monotonic()
                    while position < end:
                        block = source.read(min(1024 * 1024, end - position))
                        if not block:
                            raise RuntimeError('Unexpected archive EOF')
                        target.write(block)
                        position += len(block)
                        if time.monotonic() - last > 30:
                            print('PART %d %.1f / %.1f MiB' % (index, (position-start)/1024**2, (end-start)/1024**2), flush=True)
                            last = time.monotonic()
                print('PART %d COMPLETE' % index, flush=True)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download_part, range(8)))
        partial.replace(archive)
        verify_local(local)
    finally:
        client.close()


if __name__ == '__main__':
    main()
