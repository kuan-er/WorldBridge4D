import signal

from torch.distributed.elastic.multiprocessing import api

from worldbridge.trainer.soft_torchrun import configure_soft_cleanup


def test_elastic_cleanup_never_escalates_to_sigkill(monkeypatch):
    # Let pytest restore the original private API after this isolated check.
    monkeypatch.setattr(api, "_get_kill_signal", api._get_kill_signal)
    configure_soft_cleanup()
    assert api._get_kill_signal() == signal.SIGTERM


def test_soft_launcher_preserves_torchrun_entrypoint(monkeypatch):
    import torch.distributed.run
    from worldbridge.trainer.soft_torchrun import main

    monkeypatch.setattr(api, "_get_kill_signal", api._get_kill_signal)
    called = []
    monkeypatch.setattr(torch.distributed.run, "main", lambda: called.append(True))
    main()
    assert called == [True]
    assert api._get_kill_signal() == signal.SIGTERM
