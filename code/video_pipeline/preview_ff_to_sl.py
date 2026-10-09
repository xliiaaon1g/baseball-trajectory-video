"""Counterfactual 2D path overlay and paired Wan-Move ball conditions.

Manual FF/SL training clips supply shapes. This is a geometric preview, not
neural video editing, calibrated 3D reconstruction, or an actual SL pitch.
"""
import json
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
from train_tracknet import ROOT, OUT, dump


def main(model_track=None):
    data = json.loads((OUT / 'dataset.json').read_text())
    source, template = 'FF_02', 'SL_02'
    def points(sid):
        rows = sorted([r for r in data['rows'] if r['sample_id'] == sid], key=lambda q: q['time_seconds'])
        t = np.array([r['time_seconds'] for r in rows])
        u = (t - t[0]) / (t[-1] - t[0])
        xy = np.array([[r['x'], r['y']] for r in rows])
        return rows, t, u, xy
    ffrows, ft, fu, fxy = points(source)
    if model_track:
        ffrows = json.loads(Path(model_track).read_text())['rows']
        ft = np.array([q['time_seconds'] for q in ffrows])
        fu = (ft-ft[0])/(ft[-1]-ft[0])
        fxy = np.array([[q['x'],q['y']] for q in ffrows])
    slrows, st, su, sxy = points(template)
    u = np.linspace(0, 1, 81)
    original = np.column_stack([np.interp(u, fu, fxy[:, k]) for k in range(2)])
    # The fitted SL deviation from its endpoint chord is anchored at zero at
    # both endpoints, then placed into FF's coordinate system. No shape scaling
    # between cameras is assumed, and no physical SL validity is claimed.
    slcurve = np.column_stack([np.polyval(np.polyfit(su, sxy[:, k], 2), u) for k in range(2)])
    slresidual = slcurve - ((1-u[:, None])*slcurve[0] + u[:, None]*slcurve[-1])
    target = (1-u[:, None])*fxy[0] + u[:, None]*fxy[-1] + slresidual
    dst = OUT / ('ff_to_sl_model_preview' if model_track else 'ff_to_sl_preview')
    dst.mkdir(exist_ok=True)
    def model_xy(xy):
        # 1280x720 -> 832x468 with 6px black padding each side vertically.
        return (xy * .65 + np.array([0., 6.])).astype(np.float32)
    for name, path in [('original_ff', original), ('sl_shape_candidate', target)]:
        case = dst / name
        case.mkdir(exist_ok=True)
        np.save(case / 'tracks.npy', model_xy(path)[:, None, :])
        np.save(case / 'visibility.npy', np.ones((81, 1), dtype=bool))
        dump(case / 'metadata.json', {'frame_count': 81, 'width': 832, 'height': 480,
                                     'fps': 16, 'flight_duration_display_s': 5.,
                                     'flight_duration_source_s': float(ft[-1]-ft[0]),
                                     'time_mode': 'slow-motion; flight mapped to all 81 frames',
                                     'ball_point_only': True, 'physical_validity': 'unvalidated 2D shape',
                                     'generation_status': 'not run; runner adapter still required'})
    cap = cv2.VideoCapture(str(ROOT / ffrows[0]['source_video']))
    decoded, pts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        decoded.append(frame)
        pts.append(timestamp)
        if timestamp > ft[-1] + .03:
            break
    cap.release()
    pts = np.array(pts)
    first = decoded[int(np.argmin(abs(pts-ft[0])))]
    assert first.shape[:2] == (720, 1280)
    # Add independently tracked, nearly stationary edge/background corners
    # as scene anchors. Avoid the central pitcher/batter/catcher region.
    first_i = int(np.argmin(abs(pts-ft[0])))
    last_i = int(np.argmin(abs(pts-ft[-1])))
    gray = cv2.cvtColor(decoded[first_i], cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[:, :220] = 255
    mask[:, 1060:] = 255
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=.02,
                                      minDistance=45, mask=mask)
    if corners is None:
        raise RuntimeError('No background anchor candidates')
    initial, previous = corners.copy(), corners.copy()
    valid = np.ones(len(corners), dtype=bool)
    max_drift = np.zeros(len(corners))
    for frame in decoded[first_i+1:last_i+1]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, current, previous, None)
        backward, status_back, _ = cv2.calcOpticalFlowPyrLK(current, gray, tracked, None)
        valid &= status[:,0].astype(bool) & status_back[:,0].astype(bool)
        valid &= np.linalg.norm((backward-previous)[:,0],axis=1) < .8
        max_drift = np.maximum(max_drift,np.linalg.norm((tracked-initial)[:,0],axis=1))
        previous, gray = tracked, current
    choices = [i for i in np.argsort(max_drift) if valid[i] and max_drift[i] <= 1.5][:6]
    if len(choices) < 6:
        raise RuntimeError('Insufficient stable background anchors; need manual review')
    anchors = initial[choices,0]
    for name, ball in [('original_ff',original),('sl_shape_candidate',target)]:
        case = dst/name
        tracks = np.concatenate([np.broadcast_to(model_xy(anchors),(81,6,2)),
                                 model_xy(ball)[:,None,:]],axis=1).astype(np.float32)
        np.save(case/'tracks.npy',tracks)
        np.save(case/'visibility.npy',np.ones((81,7),dtype=bool))
        metadata = json.loads((case/'metadata.json').read_text())
        metadata.update(ball_point_only=False,ball_index=6,
                        scene_anchor_method='LK edge corners; <=1.5px source drift; <0.8px forward/back error',
                        anchors_source_px=anchors.tolist(),
                        measured_anchor_max_drift_px=max_drift[choices].tolist())
        dump(case/'metadata.json',metadata)
    first = cv2.copyMakeBorder(cv2.resize(first, (832, 468)), 6, 6, 0, 0, cv2.BORDER_CONSTANT)
    for case in ['original_ff', 'sl_shape_candidate']:
        cv2.imwrite(str(dst / case / 'first_frame.png'), first)
    frame_dir = dst / 'overlay_frames'
    frame_dir.mkdir(exist_ok=True)
    green, magenta = (80, 230, 90), (220, 90, 245)
    for i in range(81):
        t = ft[0] + u[i]*(ft[-1]-ft[0])
        frame = decoded[int(np.argmin(abs(pts-t)))].copy()
        for path, color in [(original, green), (target, magenta)]:
            cv2.polylines(frame, [np.rint(path).astype(np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.rint(path[i]).astype(int)), 8, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 10), (1040, 88), (15, 20, 25), -1)
        cv2.putText(frame, 'GREEN: '+('TrackNet' if model_track else 'manual')+' FF path | MAGENTA: SL-shape candidate', (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (240,240,240), 2, cv2.LINE_AA)
        cv2.putText(frame, 'OVERLAY ONLY | same endpoints & FF timing | no generated video', (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (240,240,240), 1, cv2.LINE_AA)
        cv2.imwrite(str(frame_dir / f'{i:03d}.png'), frame)
        if i == 40:
            # Close crop makes the measured deformation visible without
            # enlarging or replacing the actual ball in the source video.
            crop = frame[200:440, 520:780]
            cv2.imwrite(str(dst / 'comparison.png'), cv2.resize(crop, (780,720)))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '16', '-i',
                    str(frame_dir / '%03d.png'), '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst / 'overlay_preview.mp4')], check=True)
    metadata = {'source_ff': source, 'shape_template_sl': template,
                'source_path_mode': 'TrackNet inference' if model_track else 'manual labels',
                'source_label_split': 'train', 'frames': 81,
                'source_window_s': [float(ft[0]), float(ft[-1])],
                'template_window_s': [float(st[0]), float(st[-1])],
                'max_candidate_difference_px': float(np.linalg.norm(target-original, axis=1).max()),
                'method': 'quadratic SL chord residual added to FF endpoint chord',
                'common_constraints': 'same start, same end, same FF duration; SL speed not transferred',
                'limitations': ['2D geometric candidate, not physically calibrated SL',
                               'source ball still visible; no ball removal or neural generation',
                               'SL/FF camera equivalence assumed for shape preview',
                               'stationary background anchors do not preserve original player motion',
                               'visibility true is a designed condition, not observation'],
                'original_path': original.tolist(), 'candidate_path': target.tolist()}
    dump(dst / 'comparison.json', metadata)
    sample_ids = np.arange(0,81,4)
    ff_grid = np.floor(model_xy(original)[sample_ids]/8).astype(int)
    sl_grid = np.floor(model_xy(target)[sample_ids]/8).astype(int)
    dump(dst/'interface_audit.json',{
        'source':'pinned vendor create_pos_feature_map; time stride=4, spatial stride=8',
        'sampled_frames':sample_ids.tolist(),
        'differing_ball_grid_samples':int(np.any(ff_grid!=sl_grid,axis=1).sum()),
        'total_ball_samples':len(sample_ids),'ff_grids':ff_grid.tolist(),'sl_grids':sl_grid.tolist(),
        'meaning':'input condition separability only; no generation-effect claim'})
    document = '''<!doctype html><meta charset="utf-8"><title>FF→SL 球路预览</title>
<style>body{background:#101827;color:#eee;font:17px system-ui;max-width:1050px;margin:35px auto;padding:20px}video{width:100%}img{max-width:650px;width:100%}p{line-height:1.7}a{color:#80c9ff}</style>
<h1>同场景 FF → SL 球路预览</h1>
<p>绿色：FF_02 人工标注球路。紫色：SL_02 提供弯曲形状的二维候选。保留 FF 的起点、终点与飞行时间，慢放显示。</p>
<video controls loop src="overlay_preview.mp4"></video>
<p>这是球路叠加预览，原视频中的球仍然存在。尚未使用生成模型替换球，也没有验证候选属于物理上正确的 SL。捕手、投手和背景保持原视频画面。</p>
<h2>球路局部放大</h2><img src="comparison.png">
<p><a href="comparison.json">方法与轨迹数据</a></p>'''
    if model_track:
        document = document.replace('FF_02 人工标注球路','FF_02 的 TrackNet 自动捕捉球路')
    (dst / 'index.html').write_text(document)
    print(json.dumps({k:v for k,v in metadata.items() if not k.endswith('path')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-track', type=Path)
    main(parser.parse_args().model_track)
