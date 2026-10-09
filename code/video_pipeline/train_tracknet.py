"""Train/evaluate a TrackNet continuous-heatmap adaptation on reviewed FF/SL.

Fixed native-resolution ROI is derived from TRAIN only. Pitch splits stay
disjoint. No pretrained weights. No visibility claims: all retained labels
are visible. Formal accuracy is limited to this game's reviewed samples.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tracknet_preflight import ROOT, OUT, make_model, prepare


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def export(samples):
    train = np.array([[s['x'], s['y']] for s in samples if s['split'] == 'train'])
    center = (train.min(0) + train.max(0)) / 2
    size = np.ceil((train.max(0) - train.min(0) + 96) / 8).astype(int) * 8
    # Same ROI for every clip; derived once from training labels only.
    left, top = np.floor(center - size / 2).astype(int)
    width, height = size.tolist()
    cache = OUT / 'cache'
    cache.mkdir(exist_ok=True)
    rows, alignments, excluded = [], [], []
    for sid in sorted({s['sample_id'] for s in samples}):
        qs = [s for s in samples if s['sample_id'] == sid]
        cap = cv2.VideoCapture(str(ROOT / qs[0]['source_video']))
        if not cap.isOpened():
            raise RuntimeError('Cannot open ' + sid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, pts = [], []
        end = max(q['time_seconds'] for q in qs) + 1 / fps
        while True:
            ok, image = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            pts.append(t)
            frames.append(cv2.cvtColor(image[top:top+height, left:left+width], cv2.COLOR_BGR2RGB))
            if t > end:
                break
        cap.release()
        pts = np.array(pts)
        if np.any(np.diff(pts) <= 0):
            raise RuntimeError('Nonmonotonic PTS: ' + sid)
        assignments = [int(np.argmin(abs(pts - q['time_seconds']))) for q in qs]
        for q in qs:
            i = int(np.argmin(abs(pts - q['time_seconds'])))
            error = float(pts[i] - q['time_seconds'])
            if i < 2 or abs(error) > 0.6 / fps or assignments.count(i) > 1:
                excluded.append({'id': sid, 'frame_index': q['frame_index'],
                                 'reason': 'ambiguous nearest-PTS alignment',
                                 'decoded_pts': float(pts[i]), 'error_s': error})
                continue
            x, y = float(q['x'] - left), float(q['y'] - top)
            if not (0 <= x < width and 0 <= y < height):
                raise RuntimeError('Label outside train-derived ROI: ' + sid)
            name = sid + '_' + str(q['frame_index']) + '.npy'
            np.save(cache / name, np.stack([frames[j] for j in [i, i-1, i-2]]))
            rows.append({**q, 'cache': name, 'roi_x': x, 'roi_y': y,
                         'decoded_pts': float(pts[i]), 'alignment_error_s': error})
        alignments.append({'id': sid, 'fps': fps,
                           'max_alignment_error_ms': max(abs(r['alignment_error_s']) * 1000
                                                         for r in rows if r['sample_id'] == sid)})
    roi = {'left': int(left), 'top': int(top), 'width': width, 'height': height,
           'derived_from': 'train labels only', 'resize': False}
    dump(OUT / 'dataset.json', {'roi': roi, 'rows': rows, 'alignment': alignments,
                              'alignment_exclusions': excluded})
    print('exported', len(rows), 'labels; ROI', roi, flush=True)


class Balls(Dataset):
    def __init__(self, rows, roi, augment=False):
        self.rows, self.roi, self.augment = rows, roi, augment
        self.yy, self.xx = np.mgrid[:roi['height'], :roi['width']]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = np.load(OUT / 'cache' / row['cache'])
        x, y = row['roi_x'], row['roi_y']
        if self.augment:
            dx, dy = random.randint(-16, 16), random.randint(-16, 16)
            mat = np.float32([[1, 0, dx], [0, 1, dy]])
            image = np.stack([cv2.warpAffine(f, mat, (self.roi['width'], self.roi['height']),
                                            borderMode=cv2.BORDER_REFLECT_101) for f in image])
            x, y = x + dx, y + dy
        image = image.transpose(0, 3, 1, 2).reshape(9, self.roi['height'], self.roi['width'])
        image = image.astype(np.float32) / 255
        if self.augment:
            image = np.clip(image * random.uniform(0.9, 1.1), 0, 1)
        heatmap = np.exp(-((self.xx - x)**2 + (self.yy - y)**2) / (2 * 2.5**2))
        heatmap = (heatmap / heatmap.sum()).astype(np.float32)
        return torch.from_numpy(image.copy()), torch.from_numpy(heatmap), i


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for image, _, indices in loader:
            logits = model(image.to(device))[:, 0].cpu().numpy()
            for heat, index in zip(logits, indices):
                row = loader.dataset.rows[int(index)]
                y, x = np.unravel_index(heat.argmax(), heat.shape)
                error = float(np.hypot(x - row['roi_x'], y - row['roi_y']))
                predictions.append({**row, 'pred_x': float(x + loader.dataset.roi['left']),
                                    'pred_y': float(y + loader.dataset.roi['top']),
                                    'error_px': error})
    return predictions


def metrics(predictions):
    def group(ps):
        e = np.array([q['error_px'] for q in ps])
        return {'frames': len(e), 'median_px': float(np.median(e)),
                'p90_px': float(np.percentile(e, 90)), 'mean_px': float(e.mean()),
                'within_5px': float((e <= 5).mean()), 'within_10px': float((e <= 10).mean())}
    return {'overall': group(predictions),
            'by_type': {typ: group([q for q in predictions if q['pitch_type'] == typ]) for typ in ['FF', 'SL']},
            'by_pitch': {sid: group([q for q in predictions if q['sample_id'] == sid])
                         for sid in sorted({q['sample_id'] for q in predictions})}}


def train(args):
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    data = json.loads((OUT / 'dataset.json').read_text())
    rows, roi = data['rows'], data['roi']
    splits = {s: [q for q in rows if q['split'] == s] for s in ['train', 'val', 'test']}
    loaders = {s: DataLoader(Balls(v, roi, s == 'train'), batch_size=args.batch_size,
                             shuffle=s == 'train', num_workers=0) for s, v in splits.items()}
    model = make_model(heatmap=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best, history, bad = float('inf'), [], 0
    started = time.perf_counter()
    run = OUT / 'run_001'
    run.mkdir(exist_ok=True)
    if (run / 'best.pt').exists():
        raise RuntimeError('Existing run_001 checkpoint; refuse overwrite')
    config = {'device': str(device), 'seed': 101, 'epochs_max': args.epochs,
              'batch_size': args.batch_size, 'learning_rate': 1e-4, 'roi': roi,
              'architecture': 'TrackNet feature stack; Conv18 replaced by linear single heatmap',
              'loss': 'spatial cross entropy against normalized sigma=2.5px Gaussian',
              'initialization': 'random; no pretrained weights',
              'early_stop': 'validation median error, patience=5',
              'augmentation': 'joint translations +/-16px and brightness 0.9..1.1',
              'dataset_sha256': hashlib.sha256((OUT / 'dataset.json').read_bytes()).hexdigest()}
    dump(run / 'config.json', config)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total, n = 0., 0
        for batch, (image, target, _) in enumerate(loaders['train']):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)[:, 0].flatten(1)
            loss = -(target.flatten(1) * torch.log_softmax(logits, dim=1)).sum(1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(image)
            n += len(image)
            if batch % 20 == 0:
                print(f'epoch {epoch} batch {batch}/{len(loaders["train"])} loss {float(loss):.4f}', flush=True)
        val = metrics(evaluate(model, loaders['val'], device))
        entry = {'epoch': epoch, 'loss': loss_total / n, 'val': val,
                 'elapsed_s': time.perf_counter() - started}
        history.append(entry)
        dump(run / 'history.json', history)
        score = val['overall']['median_px']
        if score < best:
            best, bad = score, 0
            torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch}, run / 'best.pt')
        else:
            bad += 1
        print(json.dumps(entry), flush=True)
        if bad >= 5:
            break
    checkpoint = torch.load(run / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # Evaluate test only after validation-based model selection is finished.
    for split in ['val', 'test']:
        predictions = evaluate(model, loaders[split], device)
        dump(run / (split + '_predictions.json'), predictions)
        report = metrics(predictions)
        mean = np.array([[q['x'], q['y']] for q in splits['train']]).mean(0)
        baseline = [{**q, 'error_px': float(np.hypot(q['x'] - mean[0], q['y'] - mean[1]))}
                    for q in splits[split]]
        report['train_mean_position_baseline'] = metrics(baseline)['overall']
        report['selected_epoch'] = checkpoint['epoch']
        report['limits'] = ['same game/camera only', 'visible labeled frames only',
                            'TrackNet adaptation, not official pretrained model',
                            'frame observations correlated within pitches']
        dump(run / (split + '_metrics.json'), report)
        print(split, json.dumps(report), flush=True)
    dump(run / 'completed.json', {'elapsed_s': time.perf_counter() - started,
                                 'epochs': len(history), 'selected_epoch': checkpoint['epoch']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['export', 'train'])
    p.add_argument('--device', default='mps')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()
    if args.command == 'export':
        export(prepare())
    else:
        train(args)
