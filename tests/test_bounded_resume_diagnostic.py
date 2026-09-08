from pathlib import Path
import ast

import pytest
from worldbridge.trainer.trainer import execution_stop_step


def test_default_retains_full_horizon():
    assert execution_stop_step(150010, 160010, None) == 160010


def test_bound_does_not_extend_horizon():
    assert execution_stop_step(150010, 160010, 2) == 150012
    assert execution_stop_step(160009, 160010, 2) == 160010
    with pytest.raises(ValueError):
        execution_stop_step(150010, 160010, 0)


def test_bound_controls_loop_prefetch_and_checkpoint_not_lr():
    source = Path('src/worldbridge/trainer/trainer.py').read_text()
    assert 'for step in range(start_step, execution_end):' in source
    assert 'target_steps=execution_end,' in source
    assert 'final = completed == execution_end or bool(stop_tensor.item())' in source
    tree = ast.parse(source)
    schedules = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id in ('apply_lr_restart_schedule', 'apply_cosine_schedule')]
    assert len(schedules) == 2
    assert all('execution_end' not in ast.unparse(call) for call in schedules)
