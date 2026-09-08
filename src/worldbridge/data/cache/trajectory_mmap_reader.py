"""Opt-in, fail-closed consumer of a complete DR shared trajectory cache."""
import json
from pathlib import Path
import threading

from .native import file_sha256
from .trajectory_mmap import MEMBERS, load_clip, source_stat, verify_stream


class TrajectoryMmapReader:
    def __init__(self, root, index, source_root, complete_sha256):
        self.root = Path(root)
        self.source_root = Path(source_root)
        marker = self.root / 'bulk_complete.json'
        if not complete_sha256 or file_sha256(marker) != complete_sha256:
            raise RuntimeError('DR mmap complete marker identity mismatch')
        self.report = json.loads(marker.read_text())
        self.index_sha256 = file_sha256(index)
        rows = [json.loads(line) for line in Path(index).read_text().splitlines() if line]
        self.requests = {str(r['clip_id']): (str(r['stream']), tuple(str(f['trajectory']) for f in r['frames'])) for r in rows}
        streams = {stream for stream, _ in self.requests.values()}
        if (self.report['event'] != 'DR_MMAP_BULK_OK'
                or self.report['index_sha256'] != self.index_sha256
                or self.report['clips'] != len(rows) or len(self.requests) != len(rows)
                or self.report['streams'] != len(streams)
                or set(self.report['entries']) != streams
                or any(Path(s).name != s or s in ('.', '..') for s in streams)):
            raise RuntimeError('DR mmap full index/stream coverage mismatch')
        self._lock = threading.RLock()
        self._verified = {}

    def _identities(self, stream):
        directory = self.root / stream
        return {name: source_stat(directory/name) for name in (*MEMBERS, 'complete.json')}

    def read(self, row):
        """Hash each stream once per reader; later reads touch only clip frames.

        Verification reads full source/member bytes on first use. No NPZ decode,
        raw trajectory fallback, numerical conversion or RNG consumption occurs.
        Stat checks detect later replacement/modification of published files.
        """
        try:
            stream = str(row['stream'])
            paths = tuple(str(f['trajectory']) for f in row['frames'])
            if self.requests[str(row['clip_id'])] != (stream, paths) or len(paths) != 21:
                raise RuntimeError('DR mmap clip/frame identity mismatch')
            with self._lock:
                source = self.source_root / f'{stream}.npz'
                expected = self.report['entries'][stream]
                if source_stat(source) != expected['source_stat']:
                    raise RuntimeError('DR mmap source archive identity changed')
                before = self._identities(stream)
                if stream not in self._verified:
                    checked = verify_stream(self.root/stream, index_sha256=self.index_sha256)
                    if checked != expected or file_sha256(source) != checked['source_sha256']:
                        raise RuntimeError('DR mmap stream/source identity mismatch')
                    if before != self._identities(stream) or source_stat(source) != expected['source_stat']:
                        raise RuntimeError('DR mmap stream changed during verification')
                    self._verified[stream] = before
                elif before != self._verified[stream]:
                    raise RuntimeError('DR mmap stream changed after verification')
            result = load_clip(self.root/stream, paths, verified_report=expected)
            if self._identities(stream) != before or source_stat(source) != expected['source_stat']:
                raise RuntimeError('DR mmap stream changed during clip read')
            return result
        except (ValueError, KeyError, OSError) as exc:
            # load_geometry catches ValueError as an eligibility fallback. Cache
            # corruption/missing files must instead stop training immediately.
            raise RuntimeError(f'DR mmap input contract failed: {row.get("clip_id")}') from exc
