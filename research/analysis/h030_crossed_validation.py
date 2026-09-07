"""CPU-only boundary x GT-motion x visibility analysis of verified saved predictions.

Reuse inference, not models or training samples. Inputs may differ ONLY in checkpoint
labels/comparison declarations; prepared input identities and fixed populations match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worldbridge.evaluation.diagnostic_metrics import statistics
from worldbridge.evaluation.diagnostics import digest, json_digest, load_arrays
from worldbridge.utils.io import atomic_json


def crossed_epe(prediction, target, valid, visible, source, distance, *,
                boundary_px=2, interior_px=10, static_m=.01, large_m=.1):
    p, t = np.asarray(prediction), np.asarray(target)
    v, vis, d = np.asarray(valid, bool), np.asarray(visible, bool), np.asarray(distance)
    if p.shape != t.shape or t.ndim != 4 or t.shape[1] != 3:
        raise ValueError('XYZ must match [T,3,H,W]')
    if v.shape != (len(t), *t.shape[-2:]) or vis.shape != v.shape or d.shape != t.shape[-2:]:
        raise ValueError('fixed GT grid shape mismatch')
    if not 0 <= source < len(t) or not 0 <= static_m < large_m or not 0 <= boundary_px < interior_px:
        raise ValueError('invalid fixed stratification')
    if not np.isfinite(p).all() or not np.isfinite(t.transpose(0, 2, 3, 1)[v]).all():
        raise ValueError('non-finite XYZ; never select predictions by error')
    offdiag = (np.arange(len(t)) != source)[:, None, None]
    anchored_valid = v & v[source][None]
    anchored = v[source] & (anchored_valid & offdiag).any(axis=0)
    gt_motion = np.linalg.norm(t - t[source:source+1], axis=1)
    maximum = np.where(anchored_valid, gt_motion, 0).max(axis=0)
    motion = {
        'all': np.ones_like(d, bool),
        'motion_le_1cm': anchored & (maximum <= static_m),
        'motion_1to10cm': anchored & (maximum > static_m) & (maximum <= large_m),
        'motion_gt_10cm': anchored & (maximum > large_m),
        'motion_unavailable': ~anchored,
    }
    spatial = {'all': np.ones_like(d, bool), 'boundary': d <= boundary_px,
               'buffer2to10': (d > boundary_px) & (d <= interior_px), 'interior': d > interior_px}
    visibility = {'all': np.ones_like(v), 'visible': vis, 'occluded': ~vis}
    raw = np.linalg.norm(p - t, axis=1)
    relative = np.linalg.norm((p - p[source:source+1]) - (t - t[source:source+1]), axis=1)
    modes = {'pointmap': (raw, v & ~offdiag), 'tracking': (raw, v & offdiag),
             'displacement': (relative, anchored_valid & offdiag)}
    result = {}
    for mode, (error, population) in modes.items():
        for space, space_mask in spatial.items():
            for movement, movement_mask in motion.items():
                fixed = population & (space_mask & movement_mask)[None]
                for visibility_name, visibility_mask in visibility.items():
                    result[f'{mode}/{space}/{movement}/{visibility_name}'] = statistics(error, fixed & visibility_mask)
    return result


def rollup(rows):
    out = {}
    for metric in rows[0]['metrics']:
        values = [row['metrics'][metric] for row in rows]
        count = sum(v['count'] for v in values); total = sum(v['sum'] for v in values)
        parents = {}
        for row, value in zip(rows, values):
            if value['count']: parents.setdefault(row['parent'], []).append(value['mean'])
        out[metric] = dict(count=count, sum=total, point_weighted_mean=total/count if count else None,
                           parent_macro=float(np.mean([np.mean(x) for x in parents.values()])) if parents else None,
                           parents=len(parents), sources=sum(x['count'] > 0 for x in values))
    return out


def compare(candidate, baseline, seed):
    if candidate.keys() != baseline.keys(): raise ValueError('source comparison coverage differs')
    result = {}
    first = next(iter(candidate.values()))
    for metric in first['metrics']:
        parents = {}; empty = 0
        for key, a in candidate.items():
            b = baseline[key]
            if a['parent'] != b['parent']: raise ValueError('parent grouping differs')
            ma, mb = a['metrics'][metric], b['metrics'][metric]
            if ma['count'] != mb['count']: raise ValueError('fixed GT population changed')
            if not ma['count']:
                empty += 1
                continue
            parents.setdefault(a['parent'], []).append((ma['mean'], mb['mean']))
        means = np.asarray([np.mean(v, axis=0) for v in parents.values()])
        ca = ba = percent = delta = ci = None
        if len(parents):
            ca, ba = [float(x) for x in means.mean(axis=0)]
            delta = ca - ba; percent = 100*(ca/ba-1) if ba else None
            if len(parents) >= 2:
                rng = np.random.default_rng(seed)
                draws = (means[:, 0]-means[:, 1])[rng.integers(0, len(parents), (5000, len(parents)))].mean(axis=1)
                ci = np.quantile(draws, [.025, .975]).tolist()
        result[metric] = dict(candidate_parent_macro=ca, baseline_parent_macro=ba,
                              change_pct=percent, parent_mean_delta_m=delta, parent_bootstrap_ci95_m=ci,
                              parents=len(parents), source_pairs=len(candidate)-empty, empty_source_pairs=empty)
    return result


def analyze(inputs, comparisons, output):
    output = Path(output)
    if not output.is_relative_to('/data/WorldBridge4D-runs') or output.exists():
        raise ValueError('require a fresh persistent output directory outside Git')
    selected = {}; manifests = []; policy = data_config = None; blocked = None
    for manifest_path, root in inputs:
        manifest = json.loads(Path(manifest_path).read_text())
        spec = manifest['spec']
        if manifest['protocol'] != json_digest({'spec':spec, 'data_config':manifest['data_config']}):
            raise ValueError('manifest protocol mismatch')
        current = {k:v for k,v in spec.items() if k not in ('checkpoints', 'comparisons')}
        if policy is None:
            policy, data_config, blocked = current, manifest['data_config'], manifest['blocked']
        if policy != current or data_config != manifest['data_config'] or blocked != manifest['blocked']:
            raise ValueError('not the same fixed diagnostic protocol and missing-data population')
        if manifest['limited_gate_manifest']: raise ValueError('no training gate substitution')
        manifests.append(dict(path=manifest_path, sha256=digest(manifest_path), protocol=manifest['protocol']))
        for label, ckpt in manifest['checkpoints'].items():
            if label in selected: raise ValueError('duplicate checkpoint label')
            selected[label] = (manifest, Path(root), ckpt)
    gt_cache = {}; marker_cache = {}; records = {}; provenance = []
    for label, (manifest, root, ckpt) in selected.items():
        spec = manifest['spec']
        records[label] = {}
        for item in manifest['ready']:
            if item['cohort'] != 'validation_screen': raise ValueError('only held-out fixed validation inputs')
            marker = item['prepared']; expected_marker = item['prepared_sha256']
            if marker not in marker_cache:
                if digest(marker) != expected_marker: raise ValueError('prepared marker changed')
                metadata = json.loads(Path(marker).read_text())
                for value in metadata['files'].values():
                    if digest(value['path']) != value['sha256']: raise ValueError('prepared arrays changed')
                marker_cache[marker] = metadata
            metadata = marker_cache[marker]
            if metadata['protocol'] != item.get('prepared_protocol', manifest['protocol']):
                raise ValueError('prepared protocol changed')
            for source in item['sources']:
                key = (item['dataset'], item['index'], source)
                if key in records[label]: raise ValueError('duplicate source record in manifest')
                identity = (marker, expected_marker, source)
                if key not in gt_cache:
                    gt_cache[key] = (identity, load_arrays(metadata['files'][str(source)]))
                if gt_cache[key][0] != identity: raise ValueError('same key does not identify identical GT')
                data = gt_cache[key][1]
                path = root/'results'/label/item['cohort']/item['dataset']/f"clip-{item['index']:06d}-source-{source:02d}.json"
                record = json.loads(path.read_text())
                if record['protocol'] != manifest['protocol'] or record['checkpoint_sha256'] != ckpt['sha256']:
                    raise ValueError('prediction protocol/checkpoint mismatch')
                if record['source'] != source or record['item']['prepared_sha256'] != expected_marker:
                    raise ValueError('prediction source/input identity mismatch')
                prediction = load_arrays(record['prediction'])['xyz']
                metrics = crossed_epe(prediction, data['xyz'], data['valid'], data['visible'], source,
                                      data['depth_distance'], boundary_px=spec['boundary_distance_px'],
                                      interior_px=spec['interior_distance_px'], static_m=spec['motion_static_m'],
                                      large_m=spec['motion_large_m'])
                records[label][key] = dict(label=label, dataset=key[0], index=key[1], source=source,
                                           parent=item['parent_id'], checkpoint_sha256=ckpt['sha256'], metrics=metrics)
                provenance.append(dict(record=str(path), record_sha256=digest(path), prediction=record['prediction'],
                                       prepared=marker, prepared_sha256=expected_marker))
        print(json.dumps({'event':'crossed_checkpoint_complete', 'label':label, 'sources':len(records[label])}), flush=True)
    datasets = sorted({key[0] for rows in records.values() for key in rows})
    summary = dict(protocol=policy, manifests=manifests, blocked=blocked, checkpoints={}, comparisons={},
                   source_records=sum(len(x) for x in records.values()), CUDA_used=False,
                   caveat='four parent-balanced clips per available dataset; strata overlap; not mechanism identification or production promotion; PO remains blocked; framewise metrics not minimum-5-frame track means or GT-assisted Sim3')
    for name in datasets:
        summary['checkpoints'][name] = {label:rollup([v for k,v in rows.items() if k[0] == name]) for label, rows in records.items()}
        summary['comparisons'][name] = {}
        for ca, ba in comparisons:
            summary['comparisons'][name][f'{ca}_vs_{ba}'] = compare(
                {k:v for k,v in records[ca].items() if k[0] == name},
                {k:v for k,v in records[ba].items() if k[0] == name}, policy['seed'])
    atomic_json(output/'summary.json', summary)
    atomic_json(output/'per_source.json', [r for rows in records.values() for r in rows.values()])
    atomic_json(output/'input_provenance.json', provenance)
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({'event':'CROSSED_FIXED_VALIDATION_OK', 'output':str(output),
                      'records':summary['source_records'], 'datasets':datasets, 'blocked':len(blocked)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs=2, action='append', required=True, metavar=('MANIFEST', 'RESULT_ROOT'))
    p.add_argument('--compare', nargs=2, action='append', default=[], metavar=('CANDIDATE', 'BASELINE'))
    p.add_argument('--output', required=True)
    a = p.parse_args(); analyze(a.input, a.compare, a.output)


if __name__ == '__main__': main()
