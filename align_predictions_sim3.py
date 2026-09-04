from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/data/WorldBridge4D/src')
SOURCES = [5, 10, 15, 20]
T = 21


def fit_sim3(pred: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Deterministic Umeyama fit: aligned = scale * pred @ R.T + translation."""
    if len(pred) < 3:
        raise ValueError(f'need >=3 points for Sim(3), got {len(pred)}')
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    _, singular, Vt = np.linalg.svd(p0.T @ g0)
    R = Vt.T @ np.linalg.svd(p0.T @ g0)[0].T
    # Avoid relying on a second SVD result after numerical roundoff.
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = float(singular.sum() / max(float((p0 * p0).sum()), 1e-12))
    trans = gc - scale * (R @ pc)
    return scale, R.astype(np.float32), trans.astype(np.float32)


def inverse_preprocess(native_hw: tuple[int, int], out_hw: tuple[int, int], method: str):
    """Return raw-image coordinates for output pixel centres."""
    nh, nw = map(int, native_hw)
    oh, ow = map(int, out_hw)
    v, u = np.meshgrid(np.arange(oh, dtype=np.float64), np.arange(ow, dtype=np.float64), indexing='ij')
    if method == 'vdpm':
        resized_w = 518
        resized_h = round(nh * (518 / nw) / 14) * 14
        crop_top = (resized_h - 518) // 2 if resized_h > 518 else 0
        rx = (u + 0.5) * nw / resized_w - 0.5
        ry = (v + crop_top + 0.5) * nh / resized_h - 0.5
    elif method == '4rc':
        scale = 512 / max(nh, nw)
        resized_w, resized_h = round(nw * scale), round(nh * scale)
        cx, cy = resized_w // 2, resized_h // 2
        halfw = ((2 * cx) // 14) * 14 // 2
        halfh = ((2 * cy) // 14) * 14 // 2
        if resized_w == resized_h:
            halfh = 3 * halfw // 4
        crop_left, crop_top = cx - halfw, cy - halfh
        rx = (u + crop_left + 0.5) * nw / resized_w - 0.5
        ry = (v + crop_top + 0.5) * nh / resized_h - 0.5
    else:
        raise ValueError(method)
    gx = np.clip(np.rint((rx + 0.5) * 256 / nw - 0.5).astype(np.int64), 0, 255)
    gy = np.clip(np.rint((ry + 0.5) * 256 / nh - 0.5).astype(np.int64), 0, 255)
    return gy, gx


def make_gt_loader(dataset: str):
    if dataset == 'kubric':
        from worldbridge.training256 import KubricGeometryMmapStore, MOViF256Dataset
        metadata = Path('/dataset/data/preprocessed_256_three_dataset_v1/hot_cache/kubric_geometry_metadata')
        mmap = KubricGeometryMmapStore('/dataset/data/preprocessed_256_three_dataset_v1/hot_cache/kubric_geometry_mmap', max_open_shards=2)
        def load(index):
            p = metadata / f'geom_{index:06d}.npz'
            if not p.is_file():
                raise FileNotFoundError(p)
            # The metadata NPZ stores cameras/instances; dense depth, validity
            # and segmentation are read from the verified mmap shards.
            sample = MOViF256Dataset._load_compact_sample(p, mmap.read(index))
            return {s: MOViF256Dataset._geometry_from_sample(sample, s, True)[:2] for s in range(T)}, (512, 512)
        return load
    if dataset == 'pointodyssey':
        from worldbridge.pointodyssey import PointOdysseyDataset
        root = Path('/dataset/data/preprocessed_256_three_dataset_v1/metadata/pointodyssey_worldbridge4d_v1')
        ds = PointOdysseyDataset(root, split='validation', image_size=256)
        for row in ds.rows:
            row['source_scene'] = str(Path('/dataset/nas0/PointOdyssey') / Path(row['source_scene']).relative_to('/dataset/PointOdyssey'))
        def load(index):
            return {s: ds.source_all_targets_with_visibility(index, s)[:2] for s in range(T)}, (540, 960)
        return load
    raise ValueError(f'GT alignment unavailable for {dataset}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['vdpm', '4rc'], required=True)
    ap.add_argument('--dataset', choices=['kubric', 'pointodyssey'], required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--dest-root', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, required=True)
    args = ap.parse_args()
    src, dst = Path(args.source_root), Path(args.dest_root)
    gt_loader = make_gt_loader(args.dataset)
    failures = 0
    for i in range(args.start, args.stop):
        p = src / f'{i:06d}.npz'
        if not p.is_file():
            continue
        try:
            z = np.load(p)
            pred = np.asarray(z['xyz'], dtype=np.float32)
            pointmap = np.asarray(z['pointmap'], dtype=np.float32)
            meta = json.loads(p.with_suffix('.json').read_text())
            gy, gx = inverse_preprocess(tuple(meta['native_size']), pred.shape[2:4], args.method)
            gt_by_source, _ = gt_loader(int(z['clip_index']))
            aligned = pred.copy(); scales=[]; rotations=[]; translations=[]; epe=[]
            for qi, source in enumerate(SOURCES):
                xyz, valid = gt_by_source[source]
                gt = np.stack([xyz[t].transpose(1, 2, 0)[gy, gx] for t in range(T)])
                vm = np.stack([valid[t][gy, gx] for t in range(T)])
                mask = vm & np.isfinite(gt).all(-1) & np.isfinite(pred[qi]).all(-1)
                scale, rot, trans = fit_sim3(pred[qi][mask], gt[mask])
                aligned[qi] = scale * np.einsum('thwc,dc->thwd', pred[qi], rot) + trans
                epe.append(float(np.linalg.norm(aligned[qi][mask] - gt[mask], axis=-1).mean()))
                scales.append(scale); rotations.append(rot); translations.append(trans)
            pms=[]; gms=[]
            for t in range(T):
                xyz, valid = gt_by_source[t]
                gt = xyz[t].transpose(1, 2, 0)[gy, gx]
                mask = valid[t][gy, gx] & np.isfinite(gt).all(-1) & np.isfinite(pointmap[t]).all(-1)
                pms.append(pointmap[t][mask]); gms.append(gt[mask])
            pscale, prot, ptrans = fit_sim3(np.concatenate(pms), np.concatenate(gms))
            aligned_pointmap = pscale * np.einsum('thwc,dc->thwd', pointmap, prot) + ptrans
            dst.mkdir(parents=True, exist_ok=True)
            payload = {k: z[k] for k in z.files if k not in ('xyz','pointmap','first_frame')}
            payload.update(xyz=aligned.astype(np.float32), pointmap=aligned_pointmap.astype(np.float32),
                           first_frame=aligned[0].astype(np.float32),
                           sim3_tracking_scale=np.asarray(scales,np.float32),
                           sim3_tracking_rotation=np.stack(rotations), sim3_tracking_translation=np.stack(translations),
                           sim3_pointmap_scale=np.float32(pscale), sim3_pointmap_rotation=prot,
                           sim3_pointmap_translation=ptrans)
            np.savez_compressed(dst / p.name, **payload)
            outmeta = dict(meta, alignment='sim3', alignment_algorithm='closed_form_umeyama',
                           alignment_fit_scope='per_clip_per_source_tracking_and_per_clip_pointmap',
                           tracking_sim3_epe_m_per_source=epe, tracking_sim3_epe_m_mean=float(np.mean(epe)),
                           pointmap_sim3_fit_points=int(sum(len(x) for x in pms)), status='succeeded')
            (dst / p.with_suffix('.json').name).write_text(json.dumps(outmeta, indent=2) + '\n')
            print(json.dumps({'index':i,'status':'succeeded','tracking_epe':float(np.mean(epe))}),flush=True)
        except Exception as exc:
            failures += 1
            print(json.dumps({'index':i,'status':'failed','error':repr(exc)}),flush=True)
    print(json.dumps({'done':True,'failures':failures}),flush=True)

if __name__ == '__main__': main()
