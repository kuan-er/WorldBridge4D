import importlib.util
from pathlib import Path

import numpy as np
import pytest

from worldbridge.evaluation.diagnostic_metrics import stratified_epe

spec = importlib.util.spec_from_file_location('crossed', Path(__file__).resolve().parents[1] / 'research/analysis/h030_crossed_validation.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def sample():
    t = np.zeros((3, 3, 2, 3), np.float32)
    t[2, 0, :, 1] = .05; t[2, 0, :, 2] = .2
    v = np.ones((3, 2, 3), bool); v[1, 0, 0] = False
    vis = np.ones_like(v); vis[2] = False
    d = np.array([[0, 5, 11], [2, 10, 12]], np.float32)
    p = t + np.array([3, 4, 0], np.float32)[None, :, None, None]
    return p, t, v, vis, 1, d


def test_crossed_marginals_match_existing_fixed_diagnostic_definition():
    args = sample()
    new = module.crossed_epe(*args)
    old = stratified_epe(*args)
    for mode in ['pointmap', 'tracking']:
        for space in ['all', 'boundary', 'interior']:
            for visibility in ['all', 'visible', 'occluded']:
                assert new[f'{mode}/{space}/all/{visibility}'] == old['groups'][f'{mode}/{space}/{visibility}']
    for space in ['all', 'boundary', 'interior']:
        assert new[f'displacement/{space}/all/all'] == old['displacement_epe'][space]


def test_crossed_populations_partition_and_do_not_depend_on_prediction():
    args = sample(); a = module.crossed_epe(*args)
    b = module.crossed_epe(args[0]*7-13, *args[1:])
    assert {k:v['count'] for k,v in a.items()} == {k:v['count'] for k,v in b.items()}
    for mode in ['pointmap', 'tracking', 'displacement']:
        count = a[f'{mode}/all/all/all']['count']
        assert sum(a[f'{mode}/{space}/all/all']['count'] for space in ['boundary','buffer2to10','interior']) == count
        assert sum(a[f'{mode}/all/{motion}/all']['count'] for motion in ['motion_le_1cm','motion_1to10cm','motion_gt_10cm','motion_unavailable']) == count
        assert sum(a[f'{mode}/all/all/{vis}']['count'] for vis in ['visible','occluded']) == count
    assert a['tracking/boundary/motion_unavailable/all']['count'] == 2
    assert a['tracking/boundary/motion_le_1cm/occluded']['count'] == 1
    assert a['displacement/all/motion_unavailable/all']['count'] == 0


def test_common_bias_is_absolute_error_not_relative_displacement():
    a = module.crossed_epe(*sample())
    assert a['pointmap/all/all/all']['mean'] == 5
    assert a['tracking/all/all/all']['mean'] == 5
    assert a['displacement/all/all/all']['mean'] == pytest.approx(0, abs=3e-7)
    empty = a['pointmap/boundary/motion_1to10cm/all']
    assert empty == {'count':0, 'sum':0., 'mean':None}


@pytest.mark.parametrize('bad', ['shape','source','nonfinite_prediction','bad_threshold'])
def test_crossed_rejects_invalid_inputs_instead_of_filtering_predictions(bad):
    p,t,v,vis,s,d = sample(); kw = {}
    if bad == 'shape': d = d[:1]
    if bad == 'source': s = -1
    if bad == 'nonfinite_prediction': p[1, :, 0, 0] = np.nan  # Even invalid-source pixels cannot hide failed predictions.
    if bad == 'bad_threshold': kw['large_m'] = 0
    with pytest.raises(ValueError): module.crossed_epe(p,t,v,vis,s,d,**kw)


def row(parent, value, count):
    return {'parent':parent, 'metrics':{'m':{'mean':value if count else None,'count':count,'sum':value*count}}}


def test_paired_parent_macro_and_bootstrap_are_not_pixel_resampling():
    b = {i:row(str(i), 2, i+1) for i in range(4)}
    a = {i:row(str(i), 1, i+1) for i in range(4)}
    result = module.compare(a, b, 20260907)['m']
    assert result['change_pct'] == -50 and result['parents'] == result['source_pairs'] == 4
    assert result['parent_bootstrap_ci95_m'] == [-1, -1]
    b[0] = row('0', 2, 100)
    with pytest.raises(ValueError, match='population'): module.compare(a, b, 20260907)


def test_empty_comparison_remains_explicit_null():
    empty = {0:row('p', 0, 0)}
    result = module.compare(empty, empty, 20260907)['m']
    assert result['parents'] == result['source_pairs'] == 0 and result['empty_source_pairs'] == 1
    assert result['candidate_parent_macro'] is result['change_pct'] is result['parent_bootstrap_ci95_m'] is None
