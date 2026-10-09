"""Train one Neural ODE, select by validation, evaluate test only after selection.

No camera optimization, per-test fitting, acceleration input, or model comparison.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from export_stage1 import dump, sha256
from ode_model import BallODE, integrate

EXP = Path(__file__).resolve().parent
TRAINING = EXP/'training'


def state(row):
    return row['position_m']+row['velocity_mps']


def batch(samples, start_mode='prefix_end'):
    items = []
    for sample in samples:
        start = sample['prediction_start_row'] if start_mode == 'prefix_end' else 0
        rows = sample['rows'][start:]
        elapsed = [r['video_pts_s']-rows[0]['video_pts_s'] for r in rows]
        items.append((sample, rows, elapsed))
    length = max(len(rows) for _, rows, _ in items)
    initial, types, times, target, masks = [], [], [], [], []
    for sample, rows, elapsed in items:
        n = len(rows); padding = length-n
        initial.append(state(rows[0])); types.append([1, 0] if sample['pitch_type'] == 'FF' else [0, 1])
        times.append(elapsed+[elapsed[-1]]*padding)
        target.append([state(r) for r in rows]+[state(rows[-1])]*padding)
        masks.append([0]+[1]*(n-1)+[0]*padding)
    def tensor(x): return torch.tensor(x, dtype=torch.float64)
    return {'initial': tensor(initial), 'types': tensor(types), 'times': tensor(times),
            'target': tensor(target), 'mask': tensor(masks), 'items': items}


def losses(model, data, max_step):
    pred = integrate(model, data['initial'], data['types'], data['times'], max_step)
    residual = (pred-data['target'])/model.state_std
    mask = data['mask']; denom = mask.sum().clamp(min=1)
    position = (residual[..., :3]**2).mean(-1)
    velocity = (residual[..., 3:]**2).mean(-1)
    # Equal pitch weights despite different numbers of visible timestamps.
    counts = mask.sum(1).clamp(min=1)
    lp = ((position*mask).sum(1)/counts).mean()
    lv = ((velocity*mask).sum(1)/counts).mean()
    reg = sum((p**2).mean() for p in model.network.parameters())
    objective = lp+.25*lv+1e-6*reg
    physical_position = torch.linalg.norm(pred[..., :3]-data['target'][..., :3], dim=-1)
    physical_velocity = torch.linalg.norm(pred[..., 3:]-data['target'][..., 3:], dim=-1)
    return objective, {'position_loss': float(lp.detach()), 'velocity_loss': float(lv.detach()),
                       'position_rmse_m': float(torch.sqrt((physical_position**2*mask).sum()/denom).detach()),
                       'velocity_rmse_mps': float(torch.sqrt((physical_velocity**2*mask).sum()/denom).detach())}


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(101); np.random.seed(101); torch.use_deterministic_algorithms(True)
    prepared = TRAINING/'prepared_data.json'; data = json.loads(prepared.read_text())
    train = [s for s in data['samples'] if s['split'] == 'train']
    val = [s for s in data['samples'] if s['split'] == 'val']
    train_states = np.array([state(r) for s in train for r in s['rows']])
    mean = train_states.mean(0); std = np.maximum(train_states.std(0), [.1, 1, .1, .1, .5, .1])
    normalization = {'state_mean': mean.tolist(), 'state_std': std.tolist(),
                     'source': 'all reference frames of 13 train pitches only',
                     'sample_ids': [s['sample_id'] for s in train], 'units': ['m']*3+['m/s']*3,
                     'acceleration_output_scale_mps2': 10., 'gravity_mps2': [0, 0, -9.81]}
    dump(TRAINING/'normalization.json', normalization)
    model = BallODE(normalization)
    config = {'architecture': '8 -> 32 Tanh -> 32 Tanh -> 3 non-gravity acceleration',
              'state': ['x','y','z','vx','vy','vz'], 'pitch_encoding': {'FF':[1,0], 'SL':[0,1]},
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'solver': 'differentiable RK4, shortened last step to exact video PTS', 'max_step_s': 1/240,
              'optimizer': 'Adam', 'learning_rate': 1e-3, 'weight_decay': 0,
              'gradient_clip_norm': 10., 'minimum_validation_improvement': 1e-9,
              'output_layer_initialization': 'normal weight std 0.01, zero bias',
              'loss': 'normalized position MSE + 0.25 normalized velocity MSE + 1e-6 mean-square weight regularization',
              'per_pitch_loss_weights': 'equal', 'seed': 101, 'device': 'cpu', 'dtype': 'float64', 'torch_threads': 1,
              'batch_size_pitches': 13, 'max_epochs': args.max_epochs, 'early_stop_patience_epochs': args.patience,
              'validation_interval_epochs': 5, 'selection': 'minimum validation position+0.25 velocity loss, no weight regularization',
              'training_start': 'last reliable prefix frame', 'prediction_prefix': 'ceil(visible manual count / 3)',
              'two_dimensional_loss_used': False, 'camera_fixed': True,
              'training_samples': [s['sample_id'] for s in train], 'validation_samples': [s['sample_id'] for s in val],
              'test_samples': [s['sample_id'] for s in data['samples'] if s['split']=='test'],
              'prepared_data_sha256': sha256(prepared),
              'initial_state_source': 'Statcast reference position and velocity at last prefix frame; oracle reference input',
              'model_inputs': ['initial position_m[3]', 'initial velocity_mps[3]', 'FF/SL encoding', 'elapsed timestamps'],
              'forbidden_inputs': ['net acceleration', 'future states', 'sample_id', 'future 2D centers'],
              'limitations': 'reference states originate from whole-pitch Statcast fit; not independent 3D truth or video-only state inference',
              'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    dump(TRAINING/'config.json', config)
    tb, vb = batch(train), batch(val)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_score = float('inf'); best_weights = None; best_epoch = 0; history = []; start_time = time.perf_counter()
    for epoch in range(1, args.max_epochs+1):
        model.train(); optimizer.zero_grad()
        objective, metrics = losses(model, tb, config['max_step_s'])
        assert torch.isfinite(objective), 'Nonfinite training loss'
        objective.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        assert torch.isfinite(norm), 'Nonfinite gradient'
        optimizer.step()
        record = {'epoch': epoch, 'train_objective': float(objective.detach()), **{'train_'+k:v for k,v in metrics.items()}}
        if epoch == 1 or epoch%5 == 0:
            model.eval()
            with torch.no_grad(): _, vm = losses(model, vb, config['max_step_s'])
            score = vm['position_loss']+.25*vm['velocity_loss']
            record.update({'validation_score': score, **{'val_'+k:v for k,v in vm.items()}})
            if score < best_score-1e-9:
                best_score = score; best_epoch = epoch; best_weights = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch%25 == 0:
                print(f"epoch {epoch}: train={float(objective):.6f}, val={score:.6f}, val pos RMSE={vm['position_rmse_m']:.4f}m; {time.perf_counter()-start_time:.1f}s", flush=True)
        history.append(record)
        if epoch-best_epoch >= args.patience: break
    model.load_state_dict(best_weights); model.eval()
    elapsed = time.perf_counter()-start_time
    checkpoint = {'state_dict': model.state_dict(), 'normalization': normalization, 'config': config,
                  'selected_epoch': best_epoch, 'best_validation_score': best_score}
    torch.save(checkpoint, TRAINING/'model.pt')
    dump(TRAINING/'history.json', history)
    dump(TRAINING/'training_summary.json', {'status':'trained', 'epochs_completed':epoch, 'selected_epoch':best_epoch,
        'best_validation_score':best_score, 'elapsed_cpu_s':elapsed,
        'stop_reason':'validation patience' if epoch-best_epoch>=args.patience else 'configured epoch ceiling',
        'test_data_used_in_training_or_selection':False, 'checkpoint_sha256':sha256(TRAINING/'model.pt'),
        'normalization_train_only':True, 'config':config})
    print(f"Training complete: epoch {best_epoch}, validation {best_score:.7f}, {elapsed:.1f}s CPU", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=600)
    parser.add_argument('--patience', type=int, default=100)
    main(parser.parse_args())
