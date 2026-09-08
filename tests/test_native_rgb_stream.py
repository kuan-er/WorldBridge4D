import json
from types import SimpleNamespace

import pytest

from worldbridge.data.cache import native_rgb_stream as stream


def test_reads_only_published_filename(tmp_path):
    path = tmp_path / 'rgb_00000000.safetensors'
    path.with_suffix('.tmp').write_bytes(b'incomplete')
    with pytest.raises(TimeoutError):
        stream.wait_for_publication(path, producer_status=lambda: 'running',
                                    stopped=lambda: False, timeout=0)
    path.write_bytes(b'published')
    stream.wait_for_publication(path, producer_status=lambda: 'running',
                                stopped=lambda: False, timeout=0)


def test_wait_then_atomic_publication(tmp_path, monkeypatch):
    path = tmp_path / 'rgb.safetensors'
    slept = []
    def sleep(seconds):
        slept.append(seconds)
        path.write_bytes(b'published')
    monkeypatch.setattr(stream.time, 'sleep', sleep)
    stream.wait_for_publication(path, producer_status=lambda: 'running', stopped=lambda: False)
    assert slept == [0.5]


@pytest.mark.parametrize('status', ['failed', 'terminated', 'checkpointed_stop', 'unknown'])
def test_producer_failure_never_claims_success(tmp_path, status):
    path = tmp_path / 'rgb'
    path.write_bytes(b'existing')
    with pytest.raises(RuntimeError, match='producer ended'):
        stream.wait_for_publication(path, producer_status=lambda: status, stopped=lambda: False)


def test_missing_file_after_producer_success_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError, match='omitted'):
        stream.wait_for_publication(tmp_path / 'missing', producer_status=lambda: 'succeeded',
                                    stopped=lambda: False)


def test_stop_while_waiting(tmp_path):
    with pytest.raises(stream.RGBStreamStopped):
        stream.wait_for_publication(tmp_path / 'missing', producer_status=lambda: 'running',
                                    stopped=lambda: True)


def test_stream_stop_ends_iterator(tmp_path):
    cache = SimpleNamespace(path=lambda i: tmp_path / str(i))
    assert list(stream.iter_stream(cache, [0], producer_status=lambda: 'running',
                                   stopped=lambda: True)) == []


def test_stream_reads_cache_in_requested_order(tmp_path):
    for i in range(2):
        (tmp_path / str(i)).write_bytes(b'published')
    cache = SimpleNamespace(path=lambda i: tmp_path / str(i), iter_rgb=lambda indices: iter(indices))
    assert list(stream.iter_stream(cache, [1, 0], producer_status=lambda: 'running',
                                   stopped=lambda: False)) == [1, 0]


def test_corrupt_existing_cache_is_not_retried(tmp_path):
    path = tmp_path / '0'
    path.write_bytes(b'published_but_corrupt')
    def corrupted(_):
        raise ValueError('checksum mismatch')
    cache = SimpleNamespace(path=lambda i: path, iter_rgb=corrupted)
    with pytest.raises(ValueError, match='checksum'):
        list(stream.iter_stream(cache, [0], producer_status=lambda: 'running', stopped=lambda: False))


def test_completion_requires_success_not_merely_last_rgb():
    with pytest.raises(TimeoutError):
        stream.wait_for_success(producer_status=lambda: 'running', stopped=lambda: False, timeout=0)
    stream.wait_for_success(producer_status=lambda: 'succeeded', stopped=lambda: False, timeout=0)
    with pytest.raises(stream.RGBStreamStopped):
        stream.wait_for_success(producer_status=lambda: 'running', stopped=lambda: True)


def test_probe_pins_owner_paths_and_caches_status(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps({'status': 'running'}))
    monkeypatch.setattr(stream.subprocess, 'run', run)
    probe = stream.producer_probe('R-producer', 'owner', '/local/rgb', 'config.yaml', '/preflight')
    assert probe() == probe() == 'running'
    assert len(calls) == 1
    assert calls[0][0][-5:] == ['R-producer', 'owner', '/local/rgb', 'config.yaml', '/preflight']
    assert calls[0][1]['timeout'] == 10
