import signal
import sys

from worldbridge.data.commands.native_cache_pair import supervise


def test_two_children_complete_and_signal_handler_restored(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    commands = {name: [sys.executable, '-c',
        f'from pathlib import Path; Path({str(tmp_path / name)!r}).write_text("ok")']
        for name in ('kubric', 'dynamic_replica')}
    assert supervise(commands) == 0
    assert (tmp_path / 'kubric').read_text() == 'ok'
    assert (tmp_path / 'dynamic_replica').read_text() == 'ok'
    assert signal.getsignal(signal.SIGTERM) == before


def test_failed_worker_soft_stops_sibling(tmp_path):
    ready = tmp_path / 'ready'
    stopped = tmp_path / 'stopped'
    sibling = f'''
import signal, time
from pathlib import Path
def stop(*args):
    Path({str(stopped)!r}).write_text('SIGTERM')
    raise SystemExit(3)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).write_text('ready')
for _ in range(100): time.sleep(0.1)
'''
    failure = f'''
import time
from pathlib import Path
for _ in range(100):
    if Path({str(ready)!r}).exists(): break
    time.sleep(0.1)
raise SystemExit(7)
'''
    assert supervise({'kubric': [sys.executable, '-c', sibling],
                      'dynamic_replica': [sys.executable, '-c', failure]}) == 1
    assert stopped.read_text() == 'SIGTERM'
