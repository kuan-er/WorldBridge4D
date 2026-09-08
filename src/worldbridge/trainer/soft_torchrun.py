"""Torchrun with SIGTERM-only failure cleanup for PRL's no-SIGKILL policy.

PyTorch 2.5's elastic launcher otherwise escalates to SIGKILL after 30s even
when PRL disables SIGKILL. Keep the ordinary two-rank torchrun protocol, but
replace that escalation signal with SIGTERM and wait for worker exit. PRL
must use SIGTERM for checkpoint-first requests so torchrun forwards it to its
workers; the trainer handles SIGTERM by checkpointing at an update boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import time


def worker_health(pid: int) -> dict:
    """Read only a launcher's own worker; never signal or enumerate foreign PIDs."""
    root = Path('/proc') / str(pid)
    try:
        status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines())
        io = dict(line.split(':', 1) for line in (root / 'io').read_text().splitlines())
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'state': status['State'].strip(), 'wchan': (root / 'wchan').read_text().strip(),
                'RSS': status.get('VmRSS', '').strip(), 'HWM': status.get('VmHWM', '').strip(),
                'swap': status.get('VmSwap', '').strip(), 'threads': status['Threads'].strip(),
                'major_faults': int(fields[9]), 'read_bytes': int(io['read_bytes']),
                'oom_score': (root / 'oom_score').read_text().strip()}
    except (OSError, KeyError, ValueError, IndexError) as exc:
        return {'unavailable': type(exc).__name__}


def configure_soft_cleanup() -> None:
    from torch.distributed.elastic.multiprocessing import api

    if not callable(getattr(api, "_get_kill_signal", None)):
        raise RuntimeError("unsupported torch elastic API; cannot guarantee no SIGKILL")
    api._get_kill_signal = lambda: signal.SIGTERM
    # Elastic logs the first failure only AFTER closing its surviving workers.
    # With checkpoint-first SIGTERM that may take minutes (or be interrupted),
    # losing the initiating worker's exit code. Audit it before any cleanup.
    context = api.SubprocessContext
    if not getattr(context._poll, '_worldbridge_exit_audit', False):
        original_poll = context._poll

        def audited_poll(self):
            observed = getattr(self, '_worldbridge_observed_exits', set())
            self._worldbridge_observed_exits = observed
            now = time.monotonic()
            health_due = now >= getattr(self, '_worldbridge_next_health', 0)
            if health_due:
                self._worldbridge_next_health = now + 30
            for rank, handler in self.subprocess_handlers.items():
                code = handler.proc.poll()
                if code is not None and rank not in observed:
                    observed.add(rank)
                    print(json.dumps({
                        'event': 'rank_worker_exit', 'local_rank': rank,
                        'pid': handler.proc.pid, 'exit_code': code,
                        'signal': signal.Signals(-code).name if code < 0 else None,
                        'before_elastic_cleanup': True,
                    }), flush=True)
                elif code is None and health_due:
                    print(json.dumps({'event': 'rank_worker_health', 'local_rank': rank,
                                      'pid': handler.proc.pid, **worker_health(handler.proc.pid)}), flush=True)
            return original_poll(self)

        audited_poll._worldbridge_exit_audit = True
        context._poll = audited_poll


def main() -> None:
    from torch.distributed.run import main as torchrun

    configure_soft_cleanup()
    print(json.dumps({"event": "elastic_soft_cleanup", "escalation_signal": "SIGTERM"}), flush=True)
    torchrun()


if __name__ == "__main__":
    main()
