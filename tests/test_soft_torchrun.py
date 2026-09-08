import signal

from torch.distributed.elastic.multiprocessing import api

from worldbridge.trainer.soft_torchrun import configure_soft_cleanup


def test_elastic_cleanup_never_escalates_to_sigkill(monkeypatch):
    # Let pytest restore the original private API after this isolated check.
    monkeypatch.setattr(api, "_get_kill_signal", api._get_kill_signal)
    monkeypatch.setattr(api.SubprocessContext, '_poll', api.SubprocessContext._poll)
    configure_soft_cleanup()
    assert api._get_kill_signal() == signal.SIGTERM


def test_soft_launcher_preserves_torchrun_entrypoint(monkeypatch):
    import torch.distributed.run
    from worldbridge.trainer.soft_torchrun import main

    monkeypatch.setattr(api, "_get_kill_signal", api._get_kill_signal)
    monkeypatch.setattr(api.SubprocessContext, '_poll', api.SubprocessContext._poll)
    called = []
    monkeypatch.setattr(torch.distributed.run, "main", lambda: called.append(True))
    main()
    assert called == [True]
    assert api._get_kill_signal() == signal.SIGTERM


def test_first_worker_exit_is_logged_before_blocking_cleanup(monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    def cleanup(self):
        assert 'rank_worker_exit' in capsys.readouterr().out
        return 'original_cleanup'

    monkeypatch.setattr(api, '_get_kill_signal', api._get_kill_signal)
    monkeypatch.setattr(api.SubprocessContext, '_poll', cleanup)
    configure_soft_cleanup()
    patched = api.SubprocessContext._poll
    configure_soft_cleanup()
    assert api.SubprocessContext._poll is patched
    ctx = SimpleNamespace(subprocess_handlers={
        0: SimpleNamespace(proc=SimpleNamespace(pid=123, poll=lambda: -9)),
        1: SimpleNamespace(proc=SimpleNamespace(pid=124, poll=lambda: None)),
    })
    assert patched(ctx) == 'original_cleanup'
    assert ctx._worldbridge_observed_exits == {0}
