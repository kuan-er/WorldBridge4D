"""One-off owner-authorized cleanup; explicit terminal allowlist, never force/remove outputs.

Eligibility was checked with prl_run_inspect in session
01a0957f-a2fd-71a1-adaf-2ee558d86f2c. H030 was finished via PRL.
This is not a general Run garbage collector; other-session snapshots stay untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tarfile

ROOT = Path('/data/WorldBridge4D')
BACKUP = Path('/data/WorldBridge4D-runs/pre-main-consolidation-20260914T032858Z')
TASK = 'T-20260901153324-39dd05'
TERMINAL = {
    'R-20260913024112-5e6825': 'failed',
    'R-20260913024415-e5d9b2': 'failed',
    'R-20260913024432-40122a': 'failed',
    'R-20260913025314-a41ffc': 'failed',
    'R-20260913025327-0e4a1e': 'failed',
    'R-20260912121157-0b6dbf': 'terminated',
    'R-20260913030525-35c826': 'terminated',
    'R-20260914020237-1414b0': 'failed',
}
PROTECTED = [
    ROOT / '.pi-research/worktrees/T-20260908032232-e95da1',
    ROOT / '.pi-research/snapshots/R-20260913031335-9eddf0',
    ROOT / '.pi-research/snapshots/R-20260913031349-3f2dae',
]


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def files(path):
    result = {}
    for base, dirs, names in os.walk(path, followlinks=False):
        for name in names + [d for d in dirs if (Path(base) / d).is_symlink()]:
            p = Path(base) / name
            rel = str(p.relative_to(path))
            if '.git' in Path(rel).parts:
                continue
            result[rel] = ({'link': os.readlink(p)} if p.is_symlink() else
                           {'bytes': p.stat().st_size, 'sha256': digest(p)})
    return result


def in_use(path):
    # Only report path references, never collect command lines or signal any process.
    needle = str(path)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for item in [proc / 'cwd', proc / 'exe', *list((proc / 'fd').iterdir())]:
                try:
                    target = os.readlink(item)
                except FileNotFoundError:
                    continue
                if target == needle or target.startswith(needle + '/'):
                    return 'live_process_path_reference'
            if needle.encode() in (proc / 'cmdline').read_bytes():
                return 'live_process_argument_reference'
        except FileNotFoundError:
            continue
        except PermissionError:
            return 'process_reference_scan_permission_denied'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    assert all(p.is_dir() for p in PROTECTED)
    assert git(ROOT, 'branch', '--show-current').decode().strip() == 'main'
    assert not git(ROOT, 'status', '--porcelain')
    candidates = [ROOT / '.pi-research/worktrees' / TASK]
    candidates += [ROOT / '.pi-research/snapshots' / rid for rid in TERMINAL]
    assert len(candidates) == 9 and not set(candidates) & set(PROTECTED)
    bundle = BACKUP / 'post-main-before-cleanup.bundle'
    assert not bundle.exists(), 'one-off operation already prepared; review report before resuming'
    subprocess.run(['git', '-C', str(ROOT), 'bundle', 'create', str(bundle), '--all'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'bundle', 'verify', str(bundle)], check=True, capture_output=True)
    report = {'main_before_cleanup': git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
              'bundle_sha256': digest(bundle), 'terminal_statuses_checked_via_PRL': TERMINAL,
              'protected': list(map(str, PROTECTED)), 'removed': [], 'retained': []}
    for path in candidates:
        entry = {'path': str(path)}
        if not path.is_dir():
            entry['reason'] = 'already_absent'
            report['retained'].append(entry)
            continue
        head = git(path, 'rev-parse', 'HEAD').decode().strip()
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', head, 'main'], check=True)
        entry['head'] = head
        reason = in_use(path)
        if reason or git(path, 'status', '--porcelain'):
            entry['reason'] = reason or 'dirty_worktree'
            report['retained'].append(entry)
            continue
        before = files(path)
        archive = BACKUP / (path.name + '-complete.tar.gz')
        assert not archive.exists()
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(path, arcname=path.name, filter=lambda x: None if '.git' in Path(x.name).parts else x)
        with tarfile.open(archive) as tf:
            for rel, expected in before.items():
                member = tf.getmember(path.name + '/' + rel)
                if 'link' in expected:
                    assert member.issym() and member.linkname == expected['link']
                else:
                    assert member.isfile() and member.size == expected['bytes']
                    assert hashlib.sha256(tf.extractfile(member).read()).hexdigest() == expected['sha256']
        assert before == files(path), 'files changed during backup'
        entry.update(archive=str(archive), archive_sha256=digest(archive), files=before)
        entry['source_file_bytes'] = sum(v.get('bytes', 0) for v in before.values())
        assert git(path, 'rev-parse', 'HEAD').decode().strip() == head
        reason = in_use(path)
        if reason:
            entry['reason'] = reason
            report['retained'].append(entry)
        else:
            # No --force: Git refuses a newly dirtied/locked worktree.
            subprocess.run(['git', '-C', str(ROOT), 'worktree', 'remove', str(path)], check=True)
            report['removed'].append(entry)
        (BACKUP / 'cleanup-report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(p.is_dir() for p in PROTECTED)
    (BACKUP / 'cleanup-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'removed': [x['path'] for x in report['removed']],
                      'retained': [{'path': x['path'], 'reason': x['reason']} for x in report['retained']],
                      'report': str(BACKUP / 'cleanup-report.json')}, indent=2))


if __name__ == '__main__':
    main()
