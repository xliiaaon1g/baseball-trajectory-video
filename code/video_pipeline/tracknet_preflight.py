"""FF/SL data audit and real TrackNet-v1 training-step resource check.

Architecture follows Table II of https://arxiv.org/html/1907.03698v1.
This is a local PyTorch reimplementation, random initialization, no pretrained
checkpoint. The benchmark is NOT a trained detector or accuracy evaluation.
Labels use a 60-Hz UI index, while media uses actual presentation timestamps.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'tracknet_ff_sl'
TEST = {'FF_03', 'FF_09', 'SL_03', 'SL_08'}
VAL = {'FF_10', 'SL_10'}


def prepare():
    labels = json.loads((ROOT / 'data/tracknet_labels.json').read_text())
    annotations = json.loads((ROOT / 'data/annotations.json').read_text())
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    samples, excluded = [], []
    for row in manifest:
        if row['pitch_type'] not in {'FF', 'SL'}:
            continue
        sid = row['id']
        ann = annotations.get(sid, {})
        flags = ['reviewed', 'usable', 'continuous', 'no_swing']
        if not all(ann.get(k) is True for k in flags):
            excluded.append({'id': sid, 'reason': 'clip flags',
                             'flags': {k: ann.get(k) for k in flags}})
            continue
        for q in sorted((q for q in labels.values() if q['sample_id'] == sid),
                        key=lambda q: q['time_seconds']):
            if q.get('label_source') != 'manual':
                excluded.append({'id': sid, 'frame': q['frame_index'],
                                 'reason': 'unconfirmed candidate'})
                continue
            if not ann['release_time'] <= q['time_seconds'] <= ann['glove_pre_time']:
                excluded.append({'id': sid, 'frame': q['frame_index'],
                                 'reason': 'outside flight window'})
                continue
            if q['visible'] and not all(isinstance(q.get(k), (float, int))
                                       and math.isfinite(q[k]) for k in ['x', 'y']):
                raise ValueError('Invalid visible coordinate: ' + sid)
            samples.append({**q, 'source_video': row['video'],
                            'split': 'test' if sid in TEST else 'val' if sid in VAL else 'train'})
    counts = {}
    for split in ['train', 'val', 'test']:
        group = [s for s in samples if s['split'] == split]
        counts[split] = {'frames': len(group),
                         'pitches': len({s['sample_id'] for s in group}),
                         'by_type': dict(Counter(s['pitch_type'] for s in group)),
                         'visible': sum(s['visible'] for s in group)}
    report = {'counts': counts, 'excluded': excluded,
              'scope': ['FF', 'SL'], 'test_ids': sorted(TEST), 'val_ids': sorted(VAL),
              'limitations': ['same-game and same-camera experiment; not cross-game validation',
                              'few invisible frames; visibility evaluation insufficient',
                              'nearest decoded frame to stored UI timestamp; audit PTS before formal training']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'samples.json').write_text(json.dumps(samples, indent=2))
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(counts), flush=True)
    return samples


def make_model(heatmap=False):
    import torch.nn as nn
    layers, channels = [], 9
    # Paper padding=2 is total spatial padding: PyTorch uses 1 per side.
    blocks = [(64, 2, 'pool'), (128, 2, 'pool'), (256, 3, 'pool'),
              (512, 3, 'up'), (512, 3, 'up'), (128, 2, 'up'),
              (64, 2, None), (256, 1, None)]
    for depth, number, operation in blocks:
        for _ in range(number):
            layers.extend([nn.Conv2d(channels, depth, 3, padding=1),
                           nn.ReLU(), nn.BatchNorm2d(depth)])
            channels = depth
        if operation == 'pool':
            layers.append(nn.MaxPool2d(2))
        elif operation == 'up':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
    # Return logits: CrossEntropyLoss applies log-softmax internally.
    if heatmap:
        # Continuous spatial heatmap adaptation: retain the 17 feature
        # convolutions and replace Conv18/activation/BN with a linear head.
        layers = layers[:-3] + [nn.Conv2d(64, 1, 3, padding=1)]
    return nn.Sequential(*layers)


def benchmark(samples, device_name, steps):
    import cv2
    import numpy as np
    import torch
    torch.manual_seed(101)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process (may be sandboxed)')
    sample = next(s for s in samples if s['split'] == 'train' and s['visible'])
    cap = cv2.VideoCapture(str(ROOT / sample['source_video']))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Sequential decoding avoids timestamp-seek keyframe ambiguities. Media
    # frames are consecutive, even though labels have nominal 60-Hz indices.
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frames.append((pts, bgr))
        if pts > sample['time_seconds'] + 1 / fps:
            break
    cap.release()
    i = min(range(len(frames)), key=lambda i: abs(frames[i][0] - sample['time_seconds']))
    if i < 2 or abs(frames[i][0] - sample['time_seconds']) > 0.6 / fps:
        raise RuntimeError('Cannot align label timestamp to decoded frame')
    h, w = frames[i][1].shape[:2]
    x = np.concatenate([cv2.resize(cv2.cvtColor(frames[j][1], cv2.COLOR_BGR2RGB),
                                   (640, 360)).transpose(2, 0, 1)
                        for j in [i, i - 1, i - 2]], axis=0).copy()
    x = torch.from_numpy(x)[None].float().div(255).to(device)
    yy, xx = np.mgrid[:360, :640]
    cx, cy = sample['x'] * 640 / w, sample['y'] * 360 / h
    target = np.rint(255 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 2.5**2)))
    target = torch.from_numpy(target.astype(np.int64))[None].to(device)
    model = make_model().to(device).train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
    timings, losses = [], []
    for step in range(steps):
        sync()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all()
                                               for p in model.parameters()):
            raise RuntimeError('Nonfinite loss or gradients')
        optimizer.step()
        sync()
        timings.append(time.perf_counter() - start)
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({'step': step + 1, 'seconds': timings[-1], 'loss': losses[-1]}), flush=True)
    report = {'device': str(device), 'torch': torch.__version__, 'input_shape': list(x.shape),
              'output_shape': list(logits.shape), 'parameters': sum(p.numel() for p in model.parameters()),
              'seconds_per_step': timings, 'losses': losses, 'sample_id': sample['sample_id'],
              'label_time': sample['time_seconds'], 'decoded_pts': frames[i][0],
              'source_fps': fps, 'status': 'resource check only; repeated single sample; no accuracy claim',
              'architecture_source': 'https://arxiv.org/html/1907.03698v1#S4.T2',
              'initialization': 'PyTorch defaults; random, not pretrained'}
    (OUT / ('benchmark_' + device.type + '.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--steps', default=2, type=int)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    prepared = prepare()
    if args.benchmark:
        benchmark(prepared, args.device, args.steps)
