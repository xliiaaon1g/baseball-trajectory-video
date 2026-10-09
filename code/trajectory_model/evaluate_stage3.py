"""Freeze selected weights, replay minimal inference inputs, and export diagnostics."""
import csv
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'baseb-stage3-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from export_stage1 import dump, encode_images, probe, sha256
from ode_model import BallODE, integrate
from prepare_stage3 import project

EXP = Path(__file__).resolve().parent
TRAINING, PRED, REPORT, DELIVERY = [EXP/p for p in ['training', 'predictions', 'reports', 'delivery/stage3']]


def stats(values, units):
    a = np.asarray(values, dtype=float)
    return {'count':len(a), 'units':units, 'mean':float(a.mean()), 'median':float(np.median(a)),
            'p90':float(np.quantile(a, .9)), 'max':float(a.max()), 'rmse':float(np.sqrt(np.mean(a*a)))}


def predict(model, inference_input, step):
    # This interface cannot receive targets, acceleration, or future observations.
    initial = torch.tensor([inference_input['initial_state']], dtype=torch.float64)
    code = torch.tensor([[1., 0.] if inference_input['pitch_type']=='FF' else [0., 1.]], dtype=torch.float64)
    times = torch.tensor([inference_input['elapsed_times_s']], dtype=torch.float64)
    with torch.no_grad():
        pred = integrate(model, initial, code, times, step)[0]
        non_gravity = model.non_gravity_acceleration(pred, code.expand(len(pred), -1))
    return pred.numpy(), non_gravity.numpy()+[0,0,-9.81]


def check_solver():
    class ConstantForce(torch.nn.Module):
        def forward(self, state, code):
            a = torch.tensor([2., -1., -9.81], dtype=state.dtype).expand(len(state), -1)
            return torch.cat([state[:, 3:], a], dim=-1)
    initial = torch.tensor([[1., 2., 3., 4., 5., 6.], [2., 0., 1., -1., 3., 4.]], dtype=torch.float64)
    times = torch.tensor([[0, .0173, .072, .133, .21], [0, .0101, .05, .111, .111]], dtype=torch.float64)
    result = integrate(ConstantForce(), initial, torch.zeros((2, 2)), times)
    t = times[..., None]; a = torch.tensor([2., -1., -9.81], dtype=torch.float64)
    exact = torch.cat([initial[:, None, :3]+initial[:, None, 3:]*t+.5*a*t*t,
                       initial[:, None, 3:]+a*t], dim=-1)
    error = float((result-exact).abs().max()); assert error < 1e-11
    return {'constant_force_analytic_error_max':error, 'irregular_and_padded_times_passed':True,
            'purpose':'RK4 implementation unit check, not a model comparison'}


def main():
    torch.set_num_threads(1)
    PRED.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    data_path = TRAINING/'prepared_data.json'; data = json.loads(data_path.read_text())
    summary = json.loads((TRAINING/'training_summary.json').read_text())
    checkpoint = torch.load(TRAINING/'model.pt', map_location='cpu', weights_only=False)
    assert checkpoint['config']['prepared_data_sha256'] == sha256(data_path)
    model = BallODE(checkpoint['normalization']); model.load_state_dict(checkpoint['state_dict']); model.eval()
    assert not model.training
    for path, expected in data['source_hashes'].items(): assert sha256(Path(path)) == expected, path
    solver = check_solver()
    all_records, sample_metrics, predictions, video_checks = [], [], {}, []
    pooled = {split:{key:[] for key in ['position', 'velocity', 'predicted_observation', 'reference_observation', 'predicted_reference_projection']}
              for split in ['train', 'val', 'test']}
    for sample in data['samples']:
        sid = sample['sample_id']; start = sample['prediction_start_row']; rows = sample['rows'][start:]
        seed = rows[0]; future = rows[1:]
        initial = seed['position_m']+seed['velocity_mps']
        inference_input = {'schema_version':1, 'sample_id_metadata_only':sid,
            'initial_state':initial, 'initial_state_units':['m']*3+['m/s']*3,
            'pitch_type':sample['pitch_type'], 'elapsed_times_s':[r['video_pts_s']-seed['video_pts_s'] for r in rows],
            'initial_state_source':'Statcast reference at last reliable prefix frame; whole-pitch reference fit',
            'input_scope':'reference 3D state input, not video-only inference',
            'prediction_start_frame_metadata_only':seed['frame_index'],
            'prefix_frame_ids_metadata_only':sample['prefix_frame_ids']}
        dump(PRED/f'{sid}_input.json', inference_input)
        # Replay the saved minimal interface, not the prepared target structure.
        replay_input = json.loads((PRED/f'{sid}_input.json').read_text())
        pred, accel = predict(model, replay_input, 1/240)
        refined, _ = predict(model, replay_input, 1/480)
        assert np.all(np.isfinite(pred)) and np.all(np.isfinite(accel)) and np.array_equal(pred[0], initial)
        ref = np.array([r['position_m']+r['velocity_mps'] for r in rows])
        uv = project(sample['camera'], pred[:, :3]); ref_uv = project(sample['camera'], ref[:, :3])
        pos = np.linalg.norm(pred[1:, :3]-ref[1:, :3], axis=1)
        vel = np.linalg.norm(pred[1:, 3:]-ref[1:, 3:], axis=1)
        projected_diff = np.linalg.norm(uv[1:]-ref_uv[1:], axis=1)
        obs_ids = [i for i, r in enumerate(rows) if i>0 and r['use_for_observation_loss']]
        obs = np.array([[rows[i]['u_px'], rows[i]['v_px']] for i in obs_ids])
        obs_errors = np.linalg.norm(uv[obs_ids]-obs, axis=1)
        ref_obs_errors = np.linalg.norm(ref_uv[obs_ids]-obs, axis=1)
        depths = (pred[:, :3] @ np.array(sample['camera']['R_world_to_camera']).T+np.array(sample['camera']['T_world_to_camera_m']))[:, 2]
        ball_diameter = np.array(sample['camera']['K'])[0, 0]*.073/depths
        metrics = {'sample_id':sid, 'split':sample['split'], 'pitch_type':sample['pitch_type'],
            'prefix_manual_count':len(sample['prefix_frame_ids']), 'prediction_start_frame':seed['frame_index'],
            'future_frames':len(future), 'future_visible_observations':len(obs_ids),
            'future_missing_observations':sum(r['visibility']=='missing' for r in future),
            'prediction_horizon_s':float(inference_input['elapsed_times_s'][-1]),
            'position_error':stats(pos, 'm'), 'velocity_error':stats(vel, 'm/s'),
            'predicted_to_manual_error':stats(obs_errors, 'px'),
            'reference_to_manual_error':stats(ref_obs_errors, 'px'),
            'predicted_to_reference_projection_error':stats(projected_diff, 'px'),
            'endpoint_position_error_m':float(pos[-1]), 'endpoint_velocity_error_mps':float(vel[-1]),
            'step_halving_max_position_difference_m':float(np.linalg.norm(pred[:, :3]-refined[:, :3], axis=1).max()),
            'step_halving_max_velocity_difference_mps':float(np.linalg.norm(pred[:, 3:]-refined[:, 3:], axis=1).max()),
            'predicted_speed_range_mps':[float(x) for x in [np.linalg.norm(pred[:, 3:],axis=1).min(),np.linalg.norm(pred[:, 3:],axis=1).max()]],
            'predicted_net_acceleration_norm_range_mps2':[float(x) for x in [np.linalg.norm(accel,axis=1).min(),np.linalg.norm(accel,axis=1).max()]],
            'median_pixel_error_over_nominal_ball_diameter':float(np.median(obs_errors/ball_diameter[obs_ids])),
            'time_alignment':sample['time_alignment'], 'camera_transfer_check':sample['camera_transfer_check']}
        assert metrics['step_halving_max_position_difference_m'] < .0001
        assert metrics['step_halving_max_velocity_difference_mps'] < .0001
        sample_metrics.append(metrics)
        split = sample['split']
        for key, arr in [('position',pos),('velocity',vel),('predicted_observation',obs_errors),('reference_observation',ref_obs_errors),('predicted_reference_projection',projected_diff)]:
            pooled[split][key].extend(arr.tolist())
        records = []
        for i, r in enumerate(rows):
            record = {'sample_id':sid, 'pitch_type':sample['pitch_type'], 'split':split,
                'frame_index':r['frame_index'], 'video_pts_s':r['video_pts_s'], 'elapsed_from_seed_s':inference_input['elapsed_times_s'][i],
                'position_m':pred[i, :3].tolist(), 'velocity_mps':pred[i, 3:].tolist(),
                'net_acceleration_mps2':accel[i].tolist(), 'state_source':'reference_initial' if i==0 else 'predicted',
                'event_type':'none', 'projected_uv_px':uv[i].tolist(), 'reference_position_m':ref[i, :3].tolist(),
                'reference_velocity_mps':ref[i, 3:].tolist(), 'reference_projected_uv_px':ref_uv[i].tolist(),
                'position_error_m':float(np.linalg.norm(pred[i,:3]-ref[i,:3])),
                'velocity_error_mps':float(np.linalg.norm(pred[i,3:]-ref[i,3:])),
                'visibility':r['visibility'], 'observed_uv_px':[r['u_px'],r['v_px']] if r['use_for_observation_loss'] else None,
                'predicted_observation_error_px':float(np.linalg.norm(uv[i]-[r['u_px'],r['v_px']])) if r['use_for_observation_loss'] else None,
                'reference_observation_error_px':r['reference_reprojection_error_px'],
                'camera_segment_id':sample['camera']['camera_segment_id'],
                'used_as_prediction_target_metric':i>0}
            records.append(record); all_records.append(record)
        predictions[sid] = records
        dump(PRED/f'{sid}_prediction.json', {'input':inference_input, 'camera':sample['camera'], 'rows':records, 'metrics':metrics})
        if split in ['val', 'test']:
            directory = DELIVERY/sid; frames = directory/'prediction_frames'; frames.mkdir(exist_ok=True)
            by_frame = {r['frame_index']:r for r in records}
            sheet = Image.new('RGB', (1280, int(np.ceil(len(sample['rows'])/8))*180), 'white'); draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(sample['rows']):
                image = cv2.imread(r['source_image']); rec = by_frame.get(r['frame_index'])
                reference_uv = tuple(np.rint(r['reference_projected_uv_px']).astype(int))
                cv2.drawMarker(image, reference_uv, (0,165,255), cv2.MARKER_CROSS, 7, 1)
                if r['use_for_observation_loss']:
                    cv2.circle(image, (round(r['u_px']),round(r['v_px'])), 9, (40,220,40), 1, cv2.LINE_AA)
                if rec and rec['state_source']=='predicted':
                    cv2.drawMarker(image, tuple(np.rint(rec['projected_uv_px']).astype(int)), (230,50,230), cv2.MARKER_TILTED_CROSS, 11, 2)
                mode = 'PREFIX (reference input)' if r['frame_index']<=seed['frame_index'] else 'FUTURE (learned prediction)'
                cv2.rectangle(image, (10,10), (1250,85), (15,15,15), -1)
                cv2.putText(image, f"{sid} {split} frame={r['frame_index']} {mode}", (20,34), cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(image, 'GREEN=manual  ORANGE=Statcast reference  MAGENTA=Neural ODE future prediction', (20,58), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                if rec and rec['state_source']=='predicted':
                    error = rec['predicted_observation_error_px']
                    cv2.putText(image, f"3D reference error={rec['position_error_m']:.4f}m; manual error={error if error is not None else 'missing'}px",(20,77),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                assert cv2.imwrite(str(frames/f'{i:04d}.png'), image)
                # Fixed crop per clip includes both measured and predicted trajectory.
                coords = np.array([x['reference_projected_uv_px'] for x in sample['rows']])
                cx, cy = np.rint((coords.min(0)+coords.max(0))/2).astype(int)
                patch = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).crop((cx-52,cy-55,cx+52,cy+55)).resize((144,150))
                left, top = i%8*160+8, i//8*180+24
                sheet.paste(patch,(left,top)); draw.text((left,top-19), f"{r['frame_index']} {'PREFIX' if r['is_prefix'] else 'PREDICT'}",fill='black')
            sheet.save(directory/'prediction_contact_sheet.png')
            times = [r['clip_pts_s'] for r in sample['rows']]
            out = directory/'prediction_overlay.mp4'
            encode_images(frames,times,sample['last_frame_duration_s'],out)
            output = probe(out); tb = Fraction(output['streams'][0]['time_base'])
            pts = [float(int(f.get('pts',f['best_effort_timestamp']))*tb) for f in output['frames']]
            assert len(pts)==sample['frame_count'] and (output['streams'][0]['width'],output['streams'][0]['height'])==(1280,720)
            error = float(np.max(abs(np.array(pts)-times))); assert error<1/90000*1.1
            assert abs(float(output['streams'][0]['duration'])-sample['clip_duration_s'])<1/90000*1.1
            video_checks.append({'sample_id':sid, 'frames':len(pts), 'max_pts_error_s':error,
                'duration_s':float(output['streams'][0]['duration']), 'dimensions':[1280,720]})
        print(f"{sid} {split}: position RMSE={metrics['position_error']['rmse']:.4f}m, P90 projected/manual={metrics['predicted_to_manual_error']['p90']:.2f}px",flush=True)
    pooled_stats = {split:{key:stats(values, 'm' if key=='position' else 'm/s' if key=='velocity' else 'px')
                          for key,values in group.items()} for split,group in pooled.items()}
    dump(PRED/'metrics.json', {'samples':sample_metrics, 'pooled':pooled_stats,
        'metric_scope':'future only; seed excluded; pixel metrics exclude missing observations',
        'selection_complete_before_test_evaluation':True,
        'reference_metric_limit':'3D targets and initial states share Statcast constant acceleration fit',
        'pixel_metric_limit':'includes effective camera and time alignment error; not independent 3D truth'})
    dump(PRED/'trajectories.json', {'schema_version':1, 'state_source':'predicted after reference initial state', 'rows':all_records})
    with (PRED/'errors.csv').open('w',newline='') as stream:
        fields = ['sample_id','split','pitch_type','frame_index','elapsed_from_seed_s','state_source','position_error_m','velocity_error_mps','predicted_observation_error_px','reference_observation_error_px','visibility']
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_records)
    history = json.loads((TRAINING/'history.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].semilogy([h['epoch'] for h in history],[h['train_objective'] for h in history],label='train')
    vh = [h for h in history if 'validation_score' in h]
    axes[0].semilogy([h['epoch'] for h in vh],[h['validation_score'] for h in vh],label='validation')
    axes[0].axvline(summary['selected_epoch'],color='gray',ls='--',label='selected epoch')
    axes[0].set(xlabel='epoch',ylabel='normalized objective');axes[0].legend()
    axes[1].plot([h['epoch'] for h in vh],[h['val_position_rmse_m'] for h in vh],label='validation position RMSE')
    axes[1].set(xlabel='epoch',ylabel='position RMSE (m)');axes[1].legend();fig.tight_layout();fig.savefig(TRAINING/'training_curves.png',dpi=160);plt.close(fig)
    heldout = [s for s in data['samples'] if s['split']!='train']
    fig, axes = plt.subplots(len(heldout),2,figsize=(12,3*len(heldout)))
    for j,s in enumerate(heldout):
        recs = predictions[s['sample_id']][1:]; ts=[r['elapsed_from_seed_s'] for r in recs]
        axes[j,0].plot(ts,[r['position_error_m']*100 for r in recs],color='#a532a5')
        axes[j,0].set(title=f"{s['sample_id']} {s['split']}: 3D reference error",xlabel='elapsed from prefix end (s)',ylabel='position error (cm)')
        axes[j,1].plot(ts,[r['predicted_observation_error_px'] if r['predicted_observation_error_px'] is not None else np.nan for r in recs],label='prediction / manual',color='#a532a5')
        axes[j,1].plot(ts,[r['reference_observation_error_px'] if r['reference_observation_error_px'] is not None else np.nan for r in recs],label='reference / manual',color='#dc9000')
        axes[j,1].set(title=f"{s['sample_id']}: fixed camera projection",xlabel='elapsed from prefix end (s)',ylabel='pixel error');axes[j,1].legend()
    fig.tight_layout();fig.savefig(PRED/'heldout_error_curves.png',dpi=150);plt.close(fig)
    tests=[s for s in heldout if s['split']=='test']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,s in zip(axes.ravel(),tests):
        recs=predictions[s['sample_id']]; ref=np.array([r['reference_position_m'] for r in recs]); pp=np.array([r['position_m'] for r in recs])
        ax.plot(ref[:,1],ref[:,2],label='reference',color='#dc9000');ax.plot(pp[:,1],pp[:,2],ls='--',label='Neural ODE',color='#a532a5')
        ax.scatter(pp[0,1],pp[0,2],color='green',label='prefix end');ax.invert_xaxis()
        ax.set(title=s['sample_id'],xlabel='world y (m, toward plate)',ylabel='world z (m)');ax.legend()
    fig.tight_layout();fig.savefig(PRED/'test_trajectory_views.png',dpi=160);plt.close(fig)
    cards=''.join(f'<section><h2>{s["sample_id"]} · {s["split"]}</h2><video controls loop src="{s["sample_id"]}/prediction_overlay.mp4"></video><p><a href="{s["sample_id"]}/prediction_contact_sheet.png">逐帧检查</a></p></section>' for s in heldout)
    (DELIVERY/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阶段三：留出预测</title><style>body{font:16px system-ui;background:#f4f5f7;color:#19212b;max-width:1300px;margin:30px auto;padding:0 20px}section{background:white;border-radius:12px;padding:20px;margin:20px 0}video{width:100%}img{max-width:100%}</style><h1>阶段三：Neural ODE 后段预测</h1><p>绿色：人工球心；橙色：Statcast 三维参考投影；紫色：学习模型的后段预测。前段没有模型预测球心。</p><p>13 条训练、2 条验证、4 条测试。参考三维初态输入；相机固定。此处是球心叠加诊断视频，尚未执行外观生成。</p>'+cards+'</html>')
    validation = {'status':'passed', 'solver_check':solver, 'source_hashes_unchanged':True,
        'normalization_train_only':True, 'split_counts':{k:len([s for s in data['samples'] if s['split']==k]) for k in pooled},
        'test_pitches_used_in_training_or_selection':False, 'camera_fixed_during_training':True,
        'minimal_saved_input_replayed':True, 'no_future_teacher_forcing':True,
        'all_states_and_accelerations_finite':True, 'seed_state_exact':True,
        'step_halving_max_position_difference_m':max(m['step_halving_max_position_difference_m'] for m in sample_metrics),
        'step_halving_max_velocity_difference_mps':max(m['step_halving_max_velocity_difference_mps'] for m in sample_metrics),
        'video_checks':video_checks, 'checkpoint_sha256':sha256(TRAINING/'model.pt'),
        'model_parameter_count':checkpoint['config']['parameter_count'],
        'visual_review':'pending separate visual review record'}
    dump(REPORT/'stage3_validation.json',validation)
    report(data,summary,sample_metrics,pooled_stats)


def report(data, training, metrics, pooled):
    test=pooled['test']; valid=json.loads((TRAINING/'observation_expansion_validation.json').read_text())
    duplicates=[(s['sample_id'],r) for s in data['samples'] for r in s['redundant_same_frame_labels']]
    lines=['# 阶段三：小型 Neural ODE 运动学习','',
        '状态：本地 CPU 训练、留出投球后段预测、固定相机投影与数值检查已完成。逐帧图像检查另记于 stage3_visual_review.md。','',
        '## 数据与输入边界','',
        f"采用 9 FF + 10 SL：13 条训练、2 条验证、4 条测试，沿用原 dataset 按投球划分。{valid['frames']} 个视频帧、{valid['manual_points']} 个保留人工观测、{valid['missing_frames']} 个缺失观测。原始视频与标签的 SHA256 未改变。",
        f"旧最近帧排除的 6 个人工点已按容器时间恢复（SL_01、SL_03、SL_07 各 2 点）。{len(duplicates)} 组标注时刻落在同一解码帧，保留最后的实际人工坐标，不平均、不计作两个视频帧；冗余原标签保存在 prepared_data.json。",
        'FF_10、SL_10 的原 UI 标签仍带早期 train 字段，导出明确保留 raw_source_split，最终分组使用已有 dataset 的 val。缺失观测没有插值或伪造；三维参考可以覆盖该时间，二维观测损失/指标排除缺失帧。','',
        '每条取前 ceil(可靠人工点数/3) 个点为前段，从最后一个前段帧的六维参考状态开始预测其余帧。输入文件只保存初始位置/速度、球种、后续采样时间及说明性元数据，不输入 ax/ay/az、后段状态或后段二维球心。模型预测逐帧继续积分，没有后段真值重置。',
        '**这是参考三维初态输入的运动预测。初态和监督都来自整次投球的 Statcast 恒加速度拟合，不能称为仅凭前几帧视频恢复状态，也不能称为独立三维真值验证。**','',
        '## 相机与时间','',
        '阶段二两条训练投球得到的 FF_02 有效相机作为固定基准，各视频由静态广告/墙面 ORB + RANSAC 相似变换转移 K/R/T。未用测试球心调整相机。各片段首末背景在球路附近的最大估计漂移小于 2 px（详见 camera_transfer.json）。每条世界时间偏移只拟合前段人工点，使用 Statcast 参考几何，不用后段球心。基准相机来自完整训练投球，这在训练/测试分组内允许。',
        '三维参考本身来自整次拟合；相机仍有焦距/距离歧义，背景相似变换也不等于完整三维标定。因此二维误差包括运动、相机转移、时间配准和参考模型误差。报告同时保留“参考投影→人工”的误差，避免将全部差异归因于网络。','',
        '## 训练','',
        f"网络：8 输入（六维归一化状态 + FF/SL one-hot）→32 Tanh→32 Tanh→3 非重力加速度，共 {training['config']['parameter_count']} 个可训练参数。方程为 dp/dt=v、dv/dt=g+a_theta，g=[0,0,-9.81] m/s²，不重复添加参考净加速度中的重力。",
        '状态归一化仅使用 13 条训练投球；输出加速度尺度为固定 10 m/s²。损失为归一化位置 MSE + 0.25 速度 MSE + 1e-6 权重正则，按投球等权。本轮未加入二维投影损失，相机固定。',
        f"PyTorch {training['config']['versions']['torch']}，float64，CPU 单线程，Adam 1e-3，随机种子 101，全批次 13 条投球。RK4 最大步长 1/240 秒，每个实际 PTS 末步缩短以精确采样。实际运行 {training['epochs_completed']} 轮，验证选择第 {training['selected_epoch']} 轮，训练耗时 {training['elapsed_cpu_s']:.1f} 秒；停止原因：{training['stop_reason']}。验证集用于权重选择，测试集在权重冻结后评估，没有按测试结果重新训练。",'',
        '## 留出后段结果','',
        '下表均排除初始种子帧；3D 误差相对 Statcast 参考，二维误差只在后段可用人工观测帧计算。','',
        '| 样本 | 分组 | 后段帧/人工点 | 时长 s | 位置 RMSE cm | 终点误差 cm | 速度 RMSE m/s | 预测→人工 P90 px | 参考→人工 P90 px |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        if m['split']=='train':continue
        lines.append(f"| {m['sample_id']} | {m['split']} | {m['future_frames']}/{m['future_visible_observations']} | {m['prediction_horizon_s']:.3f} | {m['position_error']['rmse']*100:.2f} | {m['endpoint_position_error_m']*100:.2f} | {m['velocity_error']['rmse']:.3f} | {m['predicted_to_manual_error']['p90']:.2f} | {m['reference_to_manual_error']['p90']:.2f} |")
    lines += ['',f"4 条测试投球合并后：后段位置 RMSE **{test['position']['rmse']*100:.2f} cm**，速度 RMSE **{test['velocity']['rmse']:.3f} m/s**；预测投影到人工球心的中位误差 **{test['predicted_observation']['median']:.2f} px**、P90 **{test['predicted_observation']['p90']:.2f} px**。固定相机下参考投影到人工球心的中位误差 {test['reference_observation']['median']:.2f} px、P90 {test['reference_observation']['p90']:.2f} px。合并统计按帧加权；每条指标另存。",'',
        '这些结果只支持当前小数据集、FF/SL 参考轨迹演化的可行性。标签全部来自恒加速度参考，不能据此证明复杂真实空气动力学、自旋规律、跨投手或跨镜头泛化；尚未执行模型对照、初始条件编辑或生成模型。','',
        '## 检查与交付','',
        '已检查：原文件哈希、投球分组隔离、训练集归一化、保存权重和最小推理输入重放、初态精确、无后段 teacher forcing、全状态/加速度有限、相机正深度、RK4 不规则/补齐时间解析检查、步长减半结果与视频原尺寸/PTS/末帧持续时间。逐帧映射与留出叠加图另行视觉检查。','',
        '- [数据与配准明细](../training/prepared_data.json)',
        '- [训练/验证/测试划分](../training/split.json)',
        '- [相机转移与前段时间配准](../training/camera_transfer.json)',
        '- [训练配置](../training/config.json) · [归一化](../training/normalization.json) · [权重](../training/model.pt)',
        '- [训练曲线](../training/training_curves.png)',
        '- [逐投球与合并指标](../predictions/metrics.json) · [逐帧状态](../predictions/trajectories.json) · [误差 CSV](../predictions/errors.csv)',
        '- [留出误差曲线](../predictions/heldout_error_curves.png) · [测试侧视轨迹](../predictions/test_trajectory_views.png)',
        '- [验证记录](stage3_validation.json) · [视觉记录](stage3_visual_review.md)',
        '- [留出视频查看页](../delivery/stage3/index.html)','',
        '复现（本地、无需云端）：','', '```sh',
        'bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/prepare_stage3.py',
        'bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/train_stage3.py',
        'bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/evaluate_stage3.py', '```','',
        '阶段四待执行：在已检查范围内编辑球种或初始速度，使用学习模型得到两条预测球路并构造视频生成条件。','']
    (REPORT/'stage3_report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
