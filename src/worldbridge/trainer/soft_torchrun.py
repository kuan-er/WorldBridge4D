"""Torchrun with SIGTERM-only failure cleanup for PRL's no-SIGKILL policy.

PyTorch 2.5's elastic launcher otherwise escalates to SIGKILL after 30s even
when PRL disables SIGKILL. Keep the ordinary two-rank torchrun protocol, but
replace that escalation signal with SIGTERM and wait for worker exit. PRL
must use SIGTERM for checkpoint-first requests so torchrun forwards it to its
workers; the trainer handles SIGTERM by checkpointing at an update boundary.
"""
from __future__ import annotations

import json
import signal


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
