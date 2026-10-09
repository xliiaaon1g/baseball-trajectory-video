"""Expand reviewed observations and reference states without changing stages 1/2.

Background-only camera transfer; per-pitch time offset fits only prefix labels.
The 3D inputs remain Statcast reference states, not video-recovered states.
"""
import json
import math
from collections import Counter
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageDraw

import stage2_geometry as geom
from export_stage1 import dump, displayed_frame, probe, sha256

ROOT, EXP = geom.ROOT, geom.EXP
OUT = EXP / 'training'
DELIVERY = EXP / 'delivery/stage3'


def registration(a, b):
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[160:285, 20:1250] = 255
    mask[180:285, 500:820] = 0
    orb = cv2.ORB_create(nfeatures=2500)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, mask)
    if da is None or db is None:
        raise ValueError('No static background descriptors')
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .7*n.distance]
    pa = np.array([ka[m.queryIdx].pt for m in good])
    pb = np.array([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(101)
    M, ins = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2)
    if M is None or ins.sum() < 30:
        raise ValueError('Insufficient static registration inliers')
    errors = np.linalg.norm(pa @ M[:, :2].T + M[:, 2] - pb, axis=1)
    H = np.vstack([M, [0, 0, 1]])
    return H, {'matches': len(good), 'inliers': int(ins.sum()),
               'median_inlier_error_px': float(np.median(errors[ins[:, 0] == 1])),
               'similarity': H.tolist(), 'source': 'static wall ORB + RANSAC, no ball labels'}


def transferred_camera(base, H, sid):
    K = np.array(base['K']); R = np.array(base['R_world_to_camera']); T = np.array(base['T_world_to_camera_m'])
    A = H[:2, :2]; scale = np.sqrt(np.linalg.det(A))
    Q = np.eye(3); Q[:2, :2] = A/scale
    principal = A @ K[:2, 2] + H[:2, 2]
    KK = np.array([[scale*K[0, 0], 0, principal[0]], [0, scale*K[1, 1], principal[1]], [0, 0, 1]])
    RR, TT = Q @ R, Q @ T
    assert np.allclose(RR @ RR.T, np.eye(3), atol=1e-8) and abs(np.linalg.det(RR)-1) < 1e-8
    return {'sample_id': sid, 'K': KK.tolist(), 'R_world_to_camera': RR.tolist(),
            'T_world_to_camera_m': TT.tolist(), 'image_size': [1280, 720],
            'distortion': [0]*5, 'camera_segment_id': sid+'_fixed_flight',
            'calibration_source': 'stage2 FF_02 effective camera + static-background similarity',
            'base_calibration_samples': ['FF_02', 'SL_02'], 'base_calibration_split': 'train',
            'image_similarity_from_FF_02': H.tolist(), 'fitted_with_this_pitch_ball_labels': False,
            'assumptions': 'common effective camera center; square pixels; zero distortion; fixed within flight'}


def project(camera, X):
    cam = X @ np.array(camera['R_world_to_camera']).T + np.array(camera['T_world_to_camera_m'])
    assert np.all(cam[:, 2] > 0)
    uvw = cam @ np.array(camera['K']).T
    return uvw[:, :2]/uvw[:, 2:]


def fit_origin(camera, rows, ref, release_time):
    p, v, a, tr = ref
    ts = np.array([r['video_pts_s'] for r in rows]); target = np.array([[r['u_px'], r['v_px']] for r in rows])
    center = release_time - tr
    def loss(origin):
        t = ts - origin
        X = p + v*t[:, None] + .5*a*t[:, None]**2
        return float(np.sum((project(camera, X)-target)**2))
    grid = np.linspace(center-.12, center+.12, 121)
    i = int(np.argmin([loss(x) for x in grid]))
    left, right = grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(65):
        x1 = right-ratio*(right-left); x2 = left+ratio*(right-left)
        if loss(x1) < loss(x2): right = x2
        else: left = x1
    origin = (left+right)/2
    return origin, {'method': 'bounded 1D least squares; grid bracket + golden-section',
                    'fit_scope': 'prefix manual observations only',
                    'frame_ids': [r['frame_index'] for r in rows],
                    'uses_statcast_reference_geometry': True,
                    'uses_future_manual_observations': False,
                    'search_bounds_s': [float(grid[0]), float(grid[-1])],
                    'at_search_boundary': bool(abs(origin-grid[0]) < 1e-5 or abs(origin-grid[-1]) < 1e-5),
                    'prefix_rmse_px': float(np.sqrt(loss(origin)/len(rows)))}


def main():
    OUT.mkdir(exist_ok=True); DELIVERY.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'data'/p for p in ['manifest.json', 'annotations.json', 'tracknet_labels.json', 'tracknet_ff_sl/dataset.json']]
    paths += [EXP/'alignment/camera.json', EXP/'observations/observations_2d.json']
    before = {str(p): sha256(p) for p in paths}
    manifest, anns, raw, ds = [json.loads(p.read_text()) for p in paths[:4]]
    splits = {r['sample_id']: r['split'] for r in ds['rows']}
    assert all(len({r['split'] for r in ds['rows'] if r['sample_id'] == s}) == 1 for s in splits)
    base = json.loads(paths[4].read_text())['cameras']['FF_02']
    base_image = cv2.imread(str(EXP/'delivery/stage1/FF_02/frames/0000.png'))
    samples, flat, hashes = [], [], {}
    for sid, split in sorted(splits.items()):
        item = next(m for m in manifest if m['id'] == sid); ann = anns[sid]
        assert item['pitch_type'] in ['FF', 'SL']
        assert all(ann[k] for k in ['reviewed', 'usable', 'continuous', 'no_swing'])
        source = ROOT/item['video']; hashes[str(source)] = sha256(source)
        metadata = probe(source); stream = metadata['streams'][0]; tb = Fraction(stream['time_base'])
        ticks = [int(f.get('pts', f['best_effort_timestamp'])) for f in metadata['frames']]
        pts = np.array([float(t*tb) for t in ticks]); norm = pts-pts[0]
        labels = sorted([r for r in raw.values() if r['sample_id'] == sid and r['label_source'] == 'manual'
                         and ann['release_time'] <= r['time_seconds'] <= ann['glove_pre_time']], key=lambda r:r['time_seconds'])
        indexed = {}; redundant = []
        for r in labels:
            # Raw UI labels predate the final val split; exported dataset is authoritative.
            assert r['pitch_type'] == item['pitch_type'] and r['source_video'] == item['video']
            frame = displayed_frame(pts, r['time_seconds'])
            if frame in indexed:
                previous = indexed[frame]
                # Final glove_pre seek can show the same frame as the prior UI tick.
                # Keep the latest actual manual point; never average into a new label.
                redundant.append({'frame_index': frame, 'selected_ui_frame': r['frame_index'],
                    'redundant_ui_frame': previous['frame_index'], 'redundant_manual_label': previous,
                    'manual_center_difference_px': float(np.linalg.norm(np.array([r['x'], r['y']])-np.array([previous['x'], previous['y']]))),
                    'reason': 'two reviewed seeks display the same decoded frame; latest manual label retained'})
            indexed[frame] = r
        first, last = min(indexed), max(indexed)
        directory = DELIVERY/sid; frames = directory/'source_frames'; frames.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source)); selected = {}; count = 0
        while True:
            ok, image = cap.read()
            if not ok: break
            assert abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-norm[count]) < 1e-6
            if first <= count <= last:
                assert cv2.imwrite(str(frames/f'{count-first:04d}.png'), image)
                selected[count] = image
            count += 1
        cap.release(); assert count == len(pts)
        H, bg = registration(base_image, selected[first])
        Hlast, stability = registration(selected[first], selected[last])
        # Evaluate within-flight image drift near the ball region, not only affine coefficients.
        anchors = np.array([[550, 250], [650, 350], [650, 450], [700, 500]])
        drift = np.linalg.norm(anchors @ Hlast[:2, :2].T+Hlast[:2, 2]-anchors, axis=1)
        stability['max_roi_drift_px'] = float(max(drift))
        camera = transferred_camera(base, H, sid)
        rows = []
        ds_ids = {r['frame_index'] for r in ds['rows'] if r['sample_id'] == sid}
        for frame in range(first, last+1):
            label = indexed.get(frame)
            row = {'sample_id': sid, 'play_id': item['play_id'], 'pitch_type': item['pitch_type'], 'split': split,
                   'frame_index': frame, 'clip_frame_index': frame-first,
                   'video_pts_s': float(norm[frame]), 'container_pts_s': float(pts[frame]),
                   'container_pts_ticks': ticks[frame], 'time_base': stream['time_base'],
                   'clip_pts_s': float(norm[frame]-norm[first]), 'label_frame_index': None,
                   'label_time_s': None, 'u_px': None, 'v_px': None, 'visibility': 'missing',
                   'use_for_observation_loss': False, 'label_source': 'none', 'recovered_legacy_exclusion': False}
            if label:
                row.update({'label_frame_index': label['frame_index'], 'label_time_s': label['time_seconds'],
                            'u_px': label['x'], 'v_px': label['y'],
                            'visibility': 'visible' if label['visible'] else 'occluded',
                            'raw_source_split': label.get('split'), 'split_source': 'existing dataset pitch split',
                            'use_for_observation_loss': bool(label['visible']), 'label_source': 'manual',
                            'review_status': 'user_review_completed_reused',
                            'recovered_legacy_exclusion': label['frame_index'] not in ds_ids})
            rows.append(row)
        visible = [r for r in rows if r['use_for_observation_loss']]
        prefix = visible[:math.ceil(len(visible)/3)]; boundary = prefix[-1]['frame_index']
        ref = geom.reference(sid)
        origin, timing = fit_origin(camera, prefix, ref, ann['release_time']-float(pts[0]))
        p, v, a, tr = ref
        ts = np.array([r['video_pts_s']-origin for r in rows])
        X = p+v*ts[:, None]+.5*a*ts[:, None]**2; V = v+a*ts[:, None]
        uv = project(camera, X)
        for i, row in enumerate(rows):
            row.update({'t_world_s': float(ts[i]), 'position_m': X[i].tolist(), 'velocity_mps': V[i].tolist(),
                        'state_source': 'reference', 'reference_source': 'Statcast constant-net-acceleration fit',
                        'reference_projected_uv_px': uv[i].tolist(),
                        'reference_reprojection_error_px': float(np.linalg.norm(uv[i]-[row['u_px'], row['v_px']])) if row['use_for_observation_loss'] else None,
                        'is_prefix': row['frame_index'] <= boundary, 'is_future': row['frame_index'] > boundary,
                        'source_image': str(frames/f'{i:04d}.png')})
        sample = {'sample_id': sid, 'play_id': item['play_id'], 'pitch_type': item['pitch_type'], 'split': split,
                  'source_video': str(source), 'source_sha256': hashes[str(source)],
                  'frame_count': len(rows), 'manual_count': len(indexed), 'source_manual_count': len(labels),
                  'redundant_same_frame_labels': redundant, 'visible_count': len(visible),
                  'missing_count': sum(r['visibility'] == 'missing' for r in rows),
                  'recovered_manual_count': sum(r['recovered_legacy_exclusion'] for r in rows),
                  'first_decoded_frame': first, 'last_decoded_frame': last,
                  'container_start_s': float(pts[0]), 'prefix_frame_ids': [r['frame_index'] for r in prefix],
                  'prediction_start_frame': boundary, 'prediction_start_row': boundary-first,
                  'video_time_origin_s': origin, 'time_alignment': timing,
                  'background_registration': bg, 'within_clip_background': stability, 'camera': camera,
                  'camera_transfer_check': 'fixed-window supported' if max(drift) < 2 else 'background drift exceeds 2px; 2D diagnostic limited',
                  'position_at_reference_m': p.tolist(), 'velocity_at_reference_mps': v.tolist(),
                  'net_acceleration_mps2': a.tolist(), 'release_time_world_s': float(tr),
                  'reference_model': 'constant net acceleration, supplied Statcast fit; not independent 3D truth',
                  'last_frame_duration_s': float(norm[last+1]-norm[last]),
                  'clip_duration_s': float(norm[last+1]-norm[first]), 'rows': rows}
        samples.append(sample); flat.extend(rows)
        # Every mapped manual point can be checked, without altering user annotations.
        sheet = Image.new('RGB', (1280, math.ceil(len(visible)/8)*150), 'white'); draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(visible):
            image = Image.fromarray(cv2.cvtColor(selected[row['frame_index']], cv2.COLOR_BGR2RGB))
            x, y = row['u_px'], row['v_px']; cx, cy = int(x), int(y)
            left, top = n%8*160+8, n//8*150+24
            sheet.paste(image.crop((cx-24, cy-20, cx+24, cy+20)).resize((144, 120)), (left, top))
            xx, yy = left+(x-cx+24)*3, top+(y-cy+20)*3
            draw.ellipse((xx-10, yy-10, xx+10, yy+10), outline='#00cc00', width=2)
            draw.text((left, top-18), f"{sid} UI {row['label_frame_index']}", fill='black')
        sheet.save(directory/'mapping_contact_sheet.png')
        assert sha256(source) == hashes[str(source)]
        print(f"{sid} {split}: {len(rows)} frames / {len(indexed)} manual, duplicate-seeks={len(redundant)}, recovered={sample['recovered_manual_count']}, prefixRMSE={timing['prefix_rmse_px']:.2f}px, drift={max(drift):.2f}px", flush=True)
    assert before == {str(p): sha256(p) for p in paths}
    assert len(samples) == 19 and Counter(s['split'] for s in samples) == {'train': 13, 'val': 2, 'test': 4}
    dump(OUT/'prepared_data.json', {'schema_version': 1, 'stage': 3, 'samples': samples,
        'source_hashes': {**before, **hashes}, 'source_files_unchanged': True,
        'input_experiment': 'reference 3D initial state + pitch type; not video-only inference',
        'time_basis': 'normalized exact container PTS; containing-frame mapping',
        'label_source': 'existing reviewed manual labels including 6 recovered old alignment exclusions',
        'calibration_future_manual_leakage': False,
        'base_camera_scope': 'stage2 full-span calibration on two train pitches only',
        'time_alignment_scope': 'each pitch prefix only; uses Statcast reference geometry',
        'forbidden_model_inputs': ['net acceleration', 'future states', 'future manual centers', 'sample_id']})
    dump(OUT/'split.json', {k:[s['sample_id'] for s in samples if s['split']==k] for k in ['train','val','test']})
    dump(OUT/'camera_transfer.json', {s['sample_id']:{k:s[k] for k in ['camera','background_registration','within_clip_background','video_time_origin_s','time_alignment','camera_transfer_check']} for s in samples})
    dump(OUT/'observation_expansion_validation.json', {'status': 'passed', 'source_files_unchanged': True,
        'split_pitch_disjoint': True, 'samples': len(samples), 'manual_points': sum(s['manual_count'] for s in samples),
        'frames': len(flat), 'missing_frames': sum(s['missing_count'] for s in samples),
        'recovered_manual_points': sum(s['recovered_manual_count'] for s in samples),
        'no_synthetic_observations': True, 'camera_fit_uses_test_future_labels': False})


if __name__ == '__main__':
    main()
