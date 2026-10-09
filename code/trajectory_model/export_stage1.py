"""Export reviewed observations, exact video times, and overlays for stage 1.

Run with bradish_pilot/.venv/bin/python. Source annotations are read-only.
No candidate, interpolated, or model-predicted coordinate becomes an observation.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    return subprocess.check_output(args, stderr=subprocess.PIPE).decode()


def probe(path):
    return json.loads(run([FFPROBE, '-v', 'error', '-select_streams', 'v:0',
                           '-show_streams', '-show_frames', '-show_entries',
                           'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames:'
                           'frame=pts,best_effort_timestamp,pkt_duration', '-of', 'json', str(path)]))


def displayed_frame(container_pts, seek_time):
    # tracknet_label.html stores HTMLVideoElement.currentTime, a seek time
    # inside a displayed frame, not necessarily its start PTS. Match the
    # containing frame on the container timeline, not nearest normalized PTS.
    index = int(np.searchsorted(container_pts, seek_time + 1e-7, side='right') - 1)
    assert 0 <= index < len(container_pts) - 1
    assert container_pts[index] <= seek_time + 1e-7 < container_pts[index + 1]
    return index


def encode_images(directory, timestamps, last_duration_s, out):
    # Image-demuxer time base must not default to 1/25: preserve source PTS.
    # Concat durations use microseconds. Round cumulative times first so
    # per-frame rounding does not accumulate into a multi-tick PTS drift.
    microseconds = [round(t * 1_000_000) for t in timestamps]
    durations = [(b - a) / 1_000_000 for a, b in zip(microseconds, microseconds[1:])]
    durations.append(last_duration_s)
    concat = directory / 'frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for i, duration in enumerate(durations):
        lines.extend([f"file '{i:04d}.png'", 'option framerate 90000', f'duration {duration:.6f}'])
    lines.extend([f"file '{len(timestamps)-1:04d}.png'", 'option framerate 90000'])
    concat.write_text('\n'.join(lines) + '\n')
    run([FFMPEG, '-v', 'error', '-y', '-safe', '0', '-f', 'concat', '-i', str(concat),
         '-frames:v', str(len(timestamps)), '-fps_mode', 'vfr', '-enc_time_base', '1:90000',
         '-c:v', 'libx264', '-bf', '0', '-crf', '18', '-pix_fmt', 'yuv420p',
         '-bsf:v', f'setts=duration={round(last_duration_s * 90000)}:time_base=1/90000',
         '-video_track_timescale', '90000',
         '-movflags', '+faststart', str(out)])


def main(sample_ids):
    out_obs = EXPERIMENT / 'observations'
    out_reports = EXPERIMENT / 'reports'
    delivery = EXPERIMENT / 'delivery' / 'stage1'
    for path in [out_obs, out_reports, delivery]:
        path.mkdir(parents=True, exist_ok=True)
    paths = {'manifest': ROOT / 'data/manifest.json', 'annotations': ROOT / 'data/annotations.json',
             'labels': ROOT / 'data/tracknet_labels.json', 'dataset': ROOT / 'data/tracknet_ff_sl/dataset.json'}
    inputs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    before_hashes = {k: sha256(p) for k, p in paths.items()}
    all_rows, summaries, clips = [], [], []
    for sid in sample_ids:
        manifest = next(q for q in inputs['manifest'] if q['id'] == sid)
        ann = inputs['annotations'][sid]
        assert all(ann.get(k) is True for k in ['reviewed', 'usable', 'continuous', 'no_swing'])
        source = ROOT / manifest['video']
        source_hash = sha256(source)
        dataset_rows = sorted([q for q in inputs['dataset']['rows'] if q['sample_id'] == sid],
                              key=lambda q: q['decoded_pts'])
        assert dataset_rows and all(q['label_source'] == 'manual' for q in dataset_rows)
        assert len({q['split'] for q in dataset_rows}) == 1
        assert int(manifest['savant']['game_pk']) == manifest['game_pk']
        assert manifest['savant']['pitch_type'] == manifest['pitch_type']
        assert manifest['date'] == manifest['savant']['game_date']
        assert manifest['play_id'] in manifest['savant_page']
        metadata = probe(source)
        stream = metadata['streams'][0]
        tb = Fraction(stream['time_base'])
        pts_ticks = [int(q.get('pts', q['best_effort_timestamp'])) for q in metadata['frames']]
        container_pts = np.array([float(t * tb) for t in pts_ticks])
        video_pts = np.array([float((t - pts_ticks[0]) * tb) for t in pts_ticks])
        assert np.all(np.diff(video_pts) > 0)
        width, height = stream['width'], stream['height']
        assert (width, height) == (1280, 720)
        indexed = {}
        for q in dataset_rows:
            raw = inputs['labels'][f"{sid}:{q['frame_index']}"]
            assert raw['label_source'] == 'manual'
            assert raw['x'] == q['x'] and raw['y'] == q['y']
            assert q['pitch_type'] == manifest['pitch_type'] and q['game_pk'] == manifest['game_pk']
            assert q['source_video'] == manifest['video']
            assert ann['release_time'] <= q['time_seconds'] <= ann['glove_pre_time']
            legacy_i = int(np.argmin(abs(video_pts - q['decoded_pts'])))
            assert abs(video_pts[legacy_i] - q['decoded_pts']) < 1e-6
            assert legacy_i == int(np.argmin(abs(video_pts - q['time_seconds'])))
            i = displayed_frame(container_pts, q['time_seconds'])
            assert i not in indexed, f'Colliding label assignment: {sid}:{i}'
            assert abs(video_pts[legacy_i] - q['time_seconds']) < 0.6 / float(Fraction(stream['avg_frame_rate']))
            assert np.isfinite([q['x'], q['y']]).all() and 0 <= q['x'] < width and 0 <= q['y'] < height
            roi = inputs['dataset']['roi']
            assert abs(q['x'] - roi['left'] - q['roi_x']) < 1e-9
            assert abs(q['y'] - roi['top'] - q['roi_y']) < 1e-9
            indexed[i] = (q, raw)
        first, last = min(indexed), max(indexed)
        sample_dir = delivery / sid
        frames_dir = sample_dir / 'frames'
        overlay_dir = sample_dir / 'overlay_frames'
        for path in [frames_dir, overlay_dir]:
            path.mkdir(parents=True, exist_ok=True)
        sample_rows = []
        selected_frames = {}
        cap = cv2.VideoCapture(str(source))
        assert cap.isOpened()
        max_decoder_error = 0.0
        read_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i = read_count
            read_count += 1
            assert i < len(video_pts)
            max_decoder_error = max(max_decoder_error, abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - video_pts[i]))
            if not first <= i <= last:
                continue
            clip_i = i - first
            row = {'experiment_id': 'neural_ode_ball', 'sample_id': sid, 'play_id': manifest['play_id'],
                   'game_pk': manifest['game_pk'], 'pitch_type': manifest['pitch_type'],
                   'split': dataset_rows[0]['split'], 'source_video': str(source),
                   'frame_index': i, 'frame_index_basis': 'zero_based_decoded_video_frame',
                   'label_frame_index': None, 'clip_frame_index': clip_i,
                   'video_pts_s': float(video_pts[i]), 'container_pts_s': float(container_pts[i]),
                   'container_pts_ticks': pts_ticks[i], 'time_base': stream['time_base'],
                   'clip_pts_s': float(video_pts[i] - video_pts[first]),
                   'u_px': None, 'v_px': None, 'visibility': 'missing', 'label_source': 'none',
                   'review_status': 'no_manual_observation', 'use_for_observation_loss': False,
                   'label_time_s': None, 'label_time_basis': 'HTMLVideoElement.currentTime_container_timeline',
                   'alignment_error_s': None, 'label_seek_delay_s': None,
                   'legacy_decoded_pts_s': None, 'legacy_decoded_frame_index': None,
                   'frame_remap_delta': None, 'legacy_review_required': None}
            if i in indexed:
                q, raw = indexed[i]
                row.update({'label_frame_index': q['frame_index'], 'u_px': q['x'], 'v_px': q['y'],
                            'visibility': 'visible' if q['visible'] else 'occluded',
                            'label_source': q['label_source'],
                            'review_status': 'user_review_completed_reused',
                            'use_for_observation_loss': bool(q['visible']),
                            'label_time_s': q['time_seconds'],
                            'alignment_error_s': float(container_pts[i] - q['time_seconds']),
                            'label_seek_delay_s': float(q['time_seconds'] - container_pts[i]),
                            'legacy_decoded_pts_s': q['decoded_pts'],
                            'legacy_decoded_frame_index': int(np.argmin(abs(video_pts - q['decoded_pts']))),
                            'frame_remap_delta': i - int(np.argmin(abs(video_pts - q['decoded_pts']))),
                            'legacy_review_required': raw.get('review_required')})
            else:
                candidates = [q for q in inputs['labels'].values() if q['sample_id'] == sid
                              and displayed_frame(container_pts, q['time_seconds']) == i]
                row['excluded_candidate_sources'] = [q['label_source'] for q in candidates]
                row['excluded_candidate_label_frame_indices'] = [q['frame_index'] for q in candidates]
            assert cv2.imwrite(str(frames_dir / f'{clip_i:04d}.png'), frame)
            overlay = frame.copy()
            if row['visibility'] == 'visible':
                center = (round(row['u_px']), round(row['v_px']))
                cv2.circle(overlay, center, 9, (40, 230, 40), 1, cv2.LINE_AA)
                cv2.drawMarker(overlay, center, (40, 230, 40), cv2.MARKER_CROSS, 5, 1)
            caption = f"{sid} decode={i} ui={row['label_frame_index']} t={row['video_pts_s']:.6f}s {row['visibility']}"
            cv2.rectangle(overlay, (12, 12), (980, 53), (20, 20, 20), -1)
            cv2.putText(overlay, caption, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            assert cv2.imwrite(str(overlay_dir / f'{clip_i:04d}.png'), overlay)
            selected_frames[i] = frame
            sample_rows.append(row)
        cap.release()
        assert read_count == len(pts_ticks) and max_decoder_error < 1e-6
        times = [r['clip_pts_s'] for r in sample_rows]
        last_duration_s = float(video_pts[last + 1] - video_pts[last])
        expected_clip_duration_s = float(video_pts[last + 1] - video_pts[first])
        encode_images(frames_dir, times, last_duration_s, sample_dir / 'source_clip.mp4')
        encode_images(overlay_dir, times, last_duration_s, sample_dir / 'ball_center_overlay.mp4')
        encode_checks = {}
        for name in ['source_clip', 'ball_center_overlay']:
            encoded = probe(sample_dir / f'{name}.mp4')
            enc_tb = Fraction(encoded['streams'][0]['time_base'])
            enc_pts = [float(int(q.get('pts', q['best_effort_timestamp'])) * enc_tb) for q in encoded['frames']]
            assert len(enc_pts) == len(times)
            error = max(abs(a - b) for a, b in zip(enc_pts, times))
            assert error <= float(tb) * 1.1, (sid, name, error)
            assert (encoded['streams'][0]['width'], encoded['streams'][0]['height']) == (width, height)
            assert abs(float(encoded['streams'][0]['duration']) - expected_clip_duration_s) <= float(tb)
            encode_checks[name] = {'frames': len(enc_pts), 'max_pts_error_s': error,
                                   'duration_s': float(encoded['streams'][0]['duration'])}
        timeline = [{'decoded_frame_index': i, 'container_pts_ticks': pts_ticks[i],
                     'container_pts_s': float(container_pts[i]), 'video_pts_s': float(video_pts[i])}
                    for i in range(len(pts_ticks))]
        dump(out_obs / f'{sid}_video_timeline.json', {'sample_id': sid, 'time_base': stream['time_base'],
             'container_video_start_s': float(container_pts[0]), 'frames': timeline})
        dump(out_obs / f'{sid}_observations_2d.json', {'rows': sample_rows})
        context_indices = [first, (first + last) // 2, last]
        for r in sample_rows:
            if r['visibility'] == 'missing':
                context_indices.append(r['frame_index'])
        context_indices = sorted(set(context_indices))
        sheet = Image.new('RGB', (400 * len(context_indices), 360), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(context_indices):
            row = sample_rows[i-first]
            image = Image.fromarray(cv2.cvtColor(selected_frames[i], cv2.COLOR_BGR2RGB))
            cx = int(row['u_px']) if row['u_px'] is not None else 575
            cy = int(row['v_px']) if row['v_px'] is not None else 388
            patch = image.crop((cx - 60, cy - 45, cx + 60, cy + 45)).resize((360, 270))
            sheet.paste(patch, (j * 400 + 20, 70))
            draw.text((j*400+20, 12), f"{sid} decoded {i} / UI {row['label_frame_index']}", fill='black')
            draw.text((j*400+20, 31), f"t={row['video_pts_s']:.6f}s {row['visibility']}", fill='black')
            if row['u_px'] is not None:
                sx = j*400 + 20 + (row['u_px'] - (cx - 60)) * 3
                sy = 70 + (row['v_px'] - (cy - 45)) * 3
                draw.ellipse((sx-12, sy-12, sx+12, sy+12), outline='#00aa00', width=2)
        sheet.save(sample_dir / 'contact_sheet.png')
        visible_rows = [r for r in sample_rows if r['visibility'] == 'visible']
        overview = Image.new('RGB', (1280, ((len(visible_rows) + 7) // 8) * 155), 'white')
        overview_draw = ImageDraw.Draw(overview)
        for n, row in enumerate(visible_rows):
            image = Image.fromarray(cv2.cvtColor(selected_frames[row['frame_index']], cv2.COLOR_BGR2RGB))
            x, y = row['u_px'], row['v_px']
            cx, cy = int(x), int(y)
            patch = image.crop((cx-24, cy-20, cx+24, cy+20)).resize((144, 120))
            left, top = (n % 8) * 160 + 8, (n // 8) * 155 + 26
            overview.paste(patch, (left, top))
            sx, sy = left + (x-(cx-24))*3, top + (y-(cy-20))*3
            overview_draw.ellipse((sx-10, sy-10, sx+10, sy+10), outline='#00cc00', width=2)
            overview_draw.text((left, top-20), f"{sid} UI {row['label_frame_index']}", fill='black')
        overview.save(sample_dir / 'all_observations_sheet.png')
        excluded = [q for q in inputs['labels'].values() if q['sample_id'] == sid
                    and q['frame_index'] not in {r['frame_index'] for r in dataset_rows}]
        summary = {'sample_id': sid, 'play_id': manifest['play_id'], 'game_pk': manifest['game_pk'],
                   'date': manifest['date'], 'pitcher_id': manifest['savant']['pitcher'],
                   'batter_id': manifest['savant']['batter'], 'pitch_type': manifest['pitch_type'],
                   'source_video': str(source), 'source_sha256': source_hash, 'image_size': [width, height],
                   'r_frame_rate': stream['r_frame_rate'], 'avg_frame_rate': stream['avg_frame_rate'],
                   'source_decoded_frames': read_count, 'container_start_s': float(container_pts[0]),
                   'first_decoded_frame_index': first, 'last_decoded_frame_index': last,
                   'first_video_pts_s': float(video_pts[first]), 'last_video_pts_s': float(video_pts[last]),
                   'first_to_last_duration_s': float(video_pts[last] - video_pts[first]),
                   'clip_duration_s': expected_clip_duration_s,
                   'exported_frames': len(sample_rows), 'manual_observations': len(indexed),
                   'visibility_counts': dict(Counter(r['visibility'] for r in sample_rows)),
                   'split': dataset_rows[0]['split'], 'reviewed_annotations': ann,
                   'legacy_max_nearest_pts_error_ms': max(abs(q['alignment_error_s']) * 1000 for q in dataset_rows),
                   'max_label_seek_delay_ms': max(r['label_seek_delay_s'] * 1000 for r in sample_rows
                                                 if r['label_seek_delay_s'] is not None),
                   'frame_remap_deltas': dict(Counter(r['frame_remap_delta'] for r in sample_rows
                                                     if r['frame_remap_delta'] is not None)),
                   'max_ffprobe_opencv_pts_error_s': max_decoder_error,
                   'legacy_review_required_true_count': sum(r['legacy_review_required'] is True for r in sample_rows),
                   'excluded_source_labels': excluded, 'encoded_video_checks': encode_checks,
                   'identity_checks': 'manifest, label, dataset, Statcast IDs/type/date consistent; no online re-fetch',
                   'clip_review_source': 'existing reviewed/no_swing/continuous/usable annotations; user confirmed completion'}
        assert sha256(source) == source_hash
        summaries.append(summary)
        all_rows.extend(sample_rows)
        clips.append(sid)
        print(f"{sid}: {len(sample_rows)} frames, {len(indexed)} manual points, PTS checks passed", flush=True)
    assert before_hashes == {k: sha256(p) for k, p in paths.items()}
    payload = {'schema_version': 1, 'experiment_id': 'neural_ode_ball', 'stage': 1,
               'created_at': datetime.now(ZoneInfo('America/Indiana/Indianapolis')).isoformat(),
               'time_basis': 'video_pts_s is exact video PTS minus first video-frame PTS; container_pts_s is unshifted',
               'frame_index_basis': 'frame_index is zero-based decoded index; label_frame_index preserves UI index',
               'coordinate_system': 'original 1280x720 image; u right, v down, pixels',
               'label_frame_mapping': 'containing frame: container_pts <= HTML currentTime < next container_pts',
               'legacy_mapping_correction': 'old dataset used nearest normalized PTS; coordinates reused on corrected frames',
               'world_time_alignment': 'not estimated in stage 1',
               'review_basis': 'user confirmed prior human review; source flags preserved, not rewritten',
               'roi': inputs['dataset']['roi'],
               'original_to_roi': [[1, 0, -inputs['dataset']['roi']['left']],
                                   [0, 1, -inputs['dataset']['roi']['top']], [0, 0, 1]],
               'source_files': {k: {'path': str(p), 'sha256': before_hashes[k]} for k, p in paths.items()},
               'samples': summaries, 'rows': all_rows}
    dump(out_obs / 'observations_2d.json', payload)
    with (out_obs / 'observations_2d.csv').open('w', newline='') as stream:
        fields = list(all_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    dump(out_reports / 'stage1_validation.json', {'status': 'complete', 'scope': sample_ids,
         'source_files_unchanged': True, 'coordinate_roundtrip_passed': True,
         'exact_decode_and_export_pts_passed': True, 'samples': summaries})
    dump(EXPERIMENT / 'config.json', {'experiment_id': 'neural_ode_ball', 'stage1_sample_ids': sample_ids,
         'observation_source': str(paths['dataset']), 'time_basis': payload['time_basis'],
         'user_review_completed': True, 'stage2_status': 'not_started',
         'reproduce_command': 'bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/export_stage1.py'})
    lines = ['# 阶段 1：二维观测与数据接口导出', '', '状态：已完成 FF_02、SL_02 的阶段 1 导出与接口检查。', '',
             '复用用户已完成的人工检查。原视频、标注和 TrackNet 数据集均保持原样；本次没有重新训练、插值补点或执行相机配准。', '',
             '| 样本 | 原图尺寸 | 导出帧 | 人工球心 | 缺失观测 | 首末帧间时长 | 最大 seek 到帧起点间隔 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['sample_id']} | 1280×720 | {s['exported_frames']} | {s['manual_observations']} | "
                     f"{s['visibility_counts'].get('missing',0)} | {s['first_to_last_duration_s']:.6f} 秒 | "
                     f"{s['max_label_seek_delay_ms']:.3f} 毫秒 |")
    lines.extend(['', '## 时间与帧号', '',
                  '- 使用逐帧容器 PTS 与 time_base 计算实际时间，并与旧 dataset.decoded_pts 及 OpenCV 解码结果交叉核对。',
                  '- video_pts_s 从首个视频帧归零；container_pts_s 保存未归零的容器时间。FF_02 容器首帧为 0.015 秒，SL_02 为 0 秒。人工标签 time_seconds 是标注界面保存的播放器 seek 时间，使用容器时间轴匹配，不将它当作归零 PTS。',
                  '- frame_index 是实际解码帧序号；label_frame_index 保留旧界面帧号。两者不能直接互换，不以标称 60 fps 推算实际帧时间。',
                  '- 发现并修正旧导出的帧对应问题：旧脚本按最近的归零 PTS 选帧；标注界面 seek 显示的是容器时间轴上包含 currentTime 的帧。采用 container_pts <= label_time < next_container_pts；FF_02 的人工坐标映射到比旧导出早 2 帧的画面，SL_02 早 1 帧。球心坐标不变，前后帧对照及关键帧叠加用于确认对应。',
                  '- alignment_error_s = container_pts_s - label_time_s；label_seek_delay_s 为 seek 时间距帧起点的间隔，处于一个帧间隔内，并非观测或配准误差。旧映射另存 legacy_decoded_pts_s、legacy_decoded_frame_index、frame_remap_delta。',
                  '- 导出 MP4 保留飞行区间每帧的相对 PTS，原图尺寸不变，不含音轨；PNG 是解码后的无损图像，MP4 为 H.264 重编码。',
                  '- 首末帧间时长是最后一帧与第一帧的 PTS 差；视频总时长另包含末帧的正常显示时长。已单独检查末帧持续时间，避免末帧只显示一个时间刻度。',
                  '- 窗口起止按人工标签实际对应的画面保留；帧起点可早于播放器 seek 时刻，这不是重新判断出手/接球事件。片段两端不称为真实出手点或接球点。', '',
                  '## 标签来源与检查状态', '',
                  '- 53 个人工球心复用既有位置与来源，review_status 记录用户确认已完成检查；旧 review_required 单独保留，不重写源标注。',
                  '- FF_02 有 23 个保留的人工标签仍带旧 review_required=true；这是来源记录差异，按用户确认的已完成检查复用。',
                  '- FF_02 两个 CoTracker 候选位于已选可见区间之前，不纳入观测。',
                  '- SL_02 界面第 213 帧只有 interpolated 标签，对应修正后实际解码第 209 帧。该帧保留视频但标记 missing，u_px/v_px=null，不参与观测损失；missing 表示缺少人工观测，不表示球被遮挡。',
                  '- 样本 ID、play_id、视频路径、球种、日期、game_pk 与已有 Statcast 行一致；连续画面和无挥棒状态复用已有审核记录。未联网重新核对官方视频身份。', '',
                  '## 交付与复现', '',
                  '- [全部观测 JSON](../observations/observations_2d.json)',
                  '- [观测 CSV](../observations/observations_2d.csv)',
                  '- [验证明细](stage1_validation.json)',
                  '- [导出画面检查记录](stage1_visual_review.md)',
                  '- [视频查看页](../delivery/stage1/index.html)', '',
                  '```sh', 'bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/export_stage1.py', '```', '',
                  '阶段 2 待执行：单位/世界坐标核对、三维参考构造、视频与世界时间对齐、有效相机估计。当前观测已可供该阶段使用。', ''])
    (out_reports / 'stage1_report.md').write_text('\n'.join(lines))
    cards = ''.join(f'<section><h2>{sid}</h2><p>左：原飞行片段；右：人工球心叠加。均使用实际帧时间。</p>'
                    f'<div class="pair"><video controls loop src="{sid}/source_clip.mp4"></video>'
                    f'<video controls loop src="{sid}/ball_center_overlay.mp4"></video></div>'
                    f'<p><a href="{sid}/contact_sheet.png">关键帧放大检查</a> · '
                    f'<a href="{sid}/all_observations_sheet.png">全部球心叠加检查</a></p></section>' for sid in clips)
    (delivery / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>阶段 1 球心导出</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#f4f5f7;color:#19212b}'
        'section{background:white;padding:20px;margin:24px 0;border-radius:12px}.pair{display:flex;gap:16px}video{width:49%;height:auto}'
        '@media(max-width:800px){.pair{display:block}video{width:100%}}</style><h1>FF_02 / SL_02 阶段 1 导出</h1>'
        '<p>已检查标签直接复用。绿色圆环表示人工球心；missing 帧没有人工观测，不补造坐标。</p>' + cards + '</html>')
    print('Stage 1 complete:', out_reports / 'stage1_report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-ids', nargs='+', default=['FF_02', 'SL_02'])
    main(parser.parse_args().sample_ids)
