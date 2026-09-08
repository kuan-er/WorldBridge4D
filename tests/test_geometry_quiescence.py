from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
import pickle
import random

import numpy as np
import pytest
import torch

from worldbridge.trainer.batching import GeometryPrefetcher


def done(value):
    future = Future()
    future.set_result(value)
    return future


def test_quiescence_keeps_results_order_queue_and_rng():
    values = [np.arange(5) + i for i in range(4)]
    current = SimpleNamespace(geometry_futures=[done(v) for v in values[:2]])
    lookahead = SimpleNamespace(geometry_futures=[done(v) for v in values[2:]])
    prefetcher = GeometryPrefetcher.__new__(GeometryPrefetcher)
    prefetcher.pending = deque([lookahead])
    state = pickle.dumps((random.getstate(), np.random.get_state()))
    torch_state = torch.get_rng_state().clone()
    assert prefetcher.quiesce(current) == 4
    assert list(prefetcher.pending) == [lookahead]
    assert all(f.result() is value for f, value in zip(
        current.geometry_futures + lookahead.geometry_futures, values))
    assert pickle.dumps((random.getstate(), np.random.get_state())) == state
    assert torch.equal(torch_state, torch.get_rng_state())
    assert prefetcher.quiesce(current) == 4  # waiting doesn't consume inputs


def test_lookahead_contract_failure_is_fatal():
    failure = Future()
    failure.set_exception(RuntimeError('corrupt published GT'))
    prefetcher = GeometryPrefetcher.__new__(GeometryPrefetcher)
    prefetcher.pending = deque([SimpleNamespace(geometry_futures=[failure])])
    current = SimpleNamespace(geometry_futures=[done(None)])
    with pytest.raises(RuntimeError, match='corrupt published GT'):
        prefetcher.quiesce(current)
    assert len(prefetcher.pending) == 1
