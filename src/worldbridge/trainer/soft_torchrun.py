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


def main() -> None:
    from torch.distributed.run import main as torchrun

    configure_soft_cleanup()
    print(json.dumps({"event": "elastic_soft_cleanup", "escalation_signal": "SIGTERM"}), flush=True)
    torchrun()


if __name__ == "__main__":
    main()
