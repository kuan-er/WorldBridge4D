"""Execute ONLY the user-confirmed 58-inode H030/H031 checkpoint proposal.
No tensor changes, recursive deletion, symlink removal, or process control.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path('/data/WorldBridge4D-runs')
MANIFEST = ROOT/'h031-cleanup-keep155k-and-active-proposal-20260909.json'
SHA = 'd134fdaa764412e07c5a3af42b85b1ae15af17aaf0c6580e83675e9fc106bbc7'
OUT = ROOT/'h031-checkpoint-cleanup-execution-20260909'
ACTIVE = ROOT/'h031-k512-k11-mix50-prefix5-to170000-gpu23-20260909'
KEEP = [ROOT/p for p in (
 'h030-fp32-cycle0-rgb1x-step155000-reference/checkpoint-0155000.pt',
 'h031-k512-k5-original-prefetch-150012-to160010-gpu23-20260908/checkpoint-0152768.pt',
 'h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909/checkpoint-0152774.pt')]

def identity(p):
    s = p.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)

def validate_group(g, remaining=None):
    paths = g['paths'] if remaining is None else remaining
    for name in paths:
        p = Path(name)
        assert p.parent.parent == ROOT and p.parent.name.startswith(('h030-', 'h031-'))
        assert p.suffix == '.pt' and (p.name.startswith('checkpoint-') or p.name == 'latest.pt')
        assert p.resolve() == p and ACTIVE not in p.parents
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode)
        assert (s.st_dev, s.st_ino, s.st_size) == (g['device'], g['inode'], g['bytes'])
        assert s.st_nlink == len(paths), ('external hardlink or race', name, s.st_nlink, len(paths))


def process_audit(groups):
    keys = {(g['device'], g['inode']) for g in groups}
    names = {name for g in groups for name in g['paths']}
    blockers, denied = [], []
    count = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            for arg in argv:
                # Resolve path arguments, including historical handoff symlinks.
                text = os.fsdecode(arg)
                text = text.split('=', 1)[-1] if text.startswith('--') else text
                if text.startswith('/') and (text in names or str(Path(text).resolve()) in names):
                    blockers.append([proc.name, 'argv_checkpoint_reference', text])
            for fd in (proc/'fd').iterdir():
                try:
                    s = fd.stat()
                    if (s.st_dev, s.st_ino) in keys:
                        blockers.append([proc.name, 'open_fd', str(fd)])
                except (FileNotFoundError, ProcessLookupError):
                    pass
            for line in (proc/'maps').read_text().splitlines():
                f = line.split(maxsplit=5)
                major, minor = (int(v, 16) for v in f[3].split(':'))
                if (os.makedev(major, minor), int(f[4])) in keys:
                    blockers.append([proc.name, 'mmap', f[-1]])
            count += 1
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            denied.append(proc.name)
    assert not blockers and not denied, dict(blockers=blockers, unreadable_processes=denied)
    return dict(processes_checked=count, blockers=blockers, unreadable_processes=denied,
                scope='read_only_all_visible_proc_fd_maps_and_direct_path_args_not_process_control')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute-user-confirmed', action='store_true')
    args = ap.parse_args()
    raw = MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    plan = json.loads(raw); groups = plan['groups']
    assert len(groups) == 58 and Counter(g['family'] for g in groups) == {'h030':46, 'h031':12}
    assert set(plan['keep_paths']) == {str(p) for p in KEEP}
    before_keep = {str(p):identity(p) for p in KEEP}
    keep_keys = {v[:2] for v in before_keep.values()}
    keys = {(g['device'],g['inode']) for g in groups}
    assert len(keys) == 58 and not keys.intersection(keep_keys)
    assert identity(ROOT/'h031-k11-capacity-handoff-20260909/resume.pt') == identity(KEEP[2])
    assert ACTIVE.is_dir() and not ACTIVE.is_symlink()
    for g in groups:
        validate_group(g)
    # Preserve every existing noncandidate regular file in touched directories.
    candidate_paths = {name for g in groups for name in g['paths']}
    other_files = {}
    for parent in {Path(name).parent for name in candidate_paths}:
        for p in parent.iterdir():
            if str(p) not in candidate_paths and p.is_file() and not p.is_symlink():
                other_files[str(p)] = identity(p)
    audit = process_audit(groups)
    report = dict(manifest_sha256=SHA, manifest=str(MANIFEST), user_authorization='好的，可释放的都删除吧 — preceding explicit H03046 plus H03112 proposal',
                  timestamp=datetime.now(timezone.utc).isoformat(), process_audit=audit,
                  expected_unique_inodes=len(groups), expected_paths=len(candidate_paths),
                  expected_allocated_bytes=sum(g['allocated_bytes'] for g in groups),
                  keep=before_keep, seed='not_applicable_metadata_and_unlink_only',
                  python=os.sys.version, environment=dict(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    if not args.execute_user_confirmed:
        print(json.dumps(dict(event='H031_CLEANUP_DRY_RUN_OK', **report)), flush=True)
        return
    OUT.mkdir(exist_ok=False)
    (OUT/'manifest.json').write_bytes(raw)
    (OUT/'pre_delete.json').write_text(json.dumps(report, indent=2)+'\n')
    before_free = shutil.disk_usage(ROOT).free
    removed = 0
    with (OUT/'unlink_audit.jsonl').open('x') as log:
        for g in groups:
            remaining = list(g['paths'])
            while remaining:
                validate_group(g, remaining)
                p = Path(remaining[0])
                directory = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    s = os.stat(p.name, dir_fd=directory, follow_symlinks=False)
                    assert stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino,s.st_size) == (g['device'],g['inode'],g['bytes'])
                    os.unlink(p.name, dir_fd=directory)
                    os.fsync(directory)
                finally:
                    os.close(directory)
                removed += 1
                log.write(json.dumps(dict(path=str(p), device=g['device'], inode=g['inode'], timestamp=datetime.now(timezone.utc).isoformat()))+'\n')
                log.flush(); os.fsync(log.fileno())
                remaining.pop(0)
    assert all(not os.path.lexists(p) for p in candidate_paths)
    assert before_keep == {str(p):identity(p) for p in KEEP}
    assert all(identity(Path(p)) == v for p,v in other_files.items())
    after_audit = process_audit(groups)
    after_free = shutil.disk_usage(ROOT).free
    result = dict(event='H031_CLEANUP_COMPLETE', removed_paths=removed, removed_unique_inodes=len(groups),
                  unlinked_allocated_bytes=report['expected_allocated_bytes'], free_before=before_free, free_after=after_free,
                  observed_free_delta=after_free-before_free, free_delta_caveat='live filesystem concurrent writes may change free-space delta',
                  keep_unchanged=True, noncandidate_files_unchanged=len(other_files), active_directory_untouched=True,
                  post_delete_process_audit=after_audit, manifest_sha256=SHA, report_directory=str(OUT),
                  timestamp=datetime.now(timezone.utc).isoformat())
    (OUT/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
