"""Render saved held-out predictions; never changes or selects model weights."""
import csv
import html
import json
import subprocess
import cv2
import numpy as np
from train_tracknet import OUT


def main():
    run = OUT / 'run_001'
    predictions = json.loads((run / 'test_predictions.json').read_text())
    metrics = json.loads((run / 'test_metrics.json').read_text())
    roi = json.loads((run / 'config.json').read_text())['roi']
    dest = run / 'review'
    dest.mkdir(exist_ok=True)
    videos = []
    for sid in sorted({q['sample_id'] for q in predictions}):
        qs = sorted([q for q in predictions if q['sample_id'] == sid], key=lambda q:q['decoded_pts'])
        frames = dest / (sid + '_frames')
        frames.mkdir(exist_ok=True)
        worst = max(qs, key=lambda q:q['error_px'])
        for i, q in enumerate(qs):
            crop = np.load(OUT / 'cache' / q['cache'])[0]
            image = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            manual = (round(q['roi_x']), round(q['roi_y']))
            predicted = (round(q['pred_x']-roi['left']), round(q['pred_y']-roi['top']))
            cv2.circle(image, manual, 5, (50,230,80), 1, cv2.LINE_AA)
            cv2.drawMarker(image, predicted, (70,70,250), cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
            enlarged = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            w = enlarged.shape[1]
            # Width and height must be even for yuv420p.
            canvas = np.zeros((enlarged.shape[0]+96, w, 3), np.uint8)
            canvas[96:] = enlarged
            cv2.putText(canvas, f'{sid}  error {q["error_px"]:.1f} original px', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (240,240,240), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'GREEN: human  RED: model', (15, 61),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (240,240,240), 1, cv2.LINE_AA)
            cv2.putText(canvas, '3x crop | 16fps slow review | visible frames only', (15, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (200,200,200), 1, cv2.LINE_AA)
            cv2.imwrite(str(frames / f'{i:03d}.png'), canvas)
            if q['frame_index'] == worst['frame_index']:
                cv2.imwrite(str(dest / (sid + '_worst.png')), canvas)
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','16','-i',str(frames/'%03d.png'),
                        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                        str(dest/(sid+'.mp4'))],check=True)
        videos.append(sid)
    with (run / 'test_predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id','frame_index','time_seconds','decoded_pts',
                                              'x','y','pred_x','pred_y','error_px'])
        writer.writeheader()
        for q in predictions:
            writer.writerow({k:q[k] for k in writer.fieldnames})
    m = metrics['overall']
    body = f'''<h1>FF／SL TrackNet 留出投球检查</h1>
<p>本地 Apple GPU 完成训练。模型使用 TrackNet 特征网络和连续热图输出，从随机权重开始。
使用固定局部画面，保留原视频像素尺寸。按验证集选择第 {metrics['selected_epoch']} 轮权重。</p>
<p>测试集共 {m['frames']} 帧／4 次投球；误差中位数 {m['median_px']:.2f} px，P90 {m['p90_px']:.2f} px，
5 px 内比例 {100*m['within_5px']:.1f}%。像素误差指原视频尺寸，绿色为人工标注，红色为模型输出。</p>
<p>同一场比赛、相似机位；这里只测试人工标为可见的帧，尚未测试不可见／无球判定或其他比赛泛化。</p>
<table><tr><th>球种</th><th>帧数</th><th>中位误差</th><th>5 px 内比例</th></tr>'''
    for typ, q in metrics['by_type'].items():
        body += f'<tr><td>{typ}</td><td>{q["frames"]}</td><td>{q["median_px"]:.2f} px</td><td>{100*q["within_5px"]:.1f}%</td></tr>'
    body += '</table>'
    for sid in videos:
        body += f'<h2>{html.escape(sid)}</h2><video controls loop src="{sid}.mp4"></video><details><summary>该投球误差最大的帧</summary><img src="{sid}_worst.png"></details>'
    body += '<p>两帧明显失败仍保留在指标中：SL_03 的 27.1px 和 SL_08 的 57.2px 误差。视觉检查显示模型误追了捕手白色服装。</p>'
    body += '<p><a href="../../ff_to_sl_model_preview/index.html">查看模型捕捉的同场景 FF→SL 球路候选</a> · <a href="../test_metrics.json">完整测试指标</a> · <a href="../test_predictions.csv">逐帧预测</a></p>'
    (dest / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TrackNet 训练结果</title><style>body{font:17px system-ui;background:#101827;color:#eee;max-width:960px;margin:35px auto;padding:20px}p{line-height:1.7}video,img{max-width:696px;width:100%}a{color:#7acfff}td,th{padding:12px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #445}</style>'+body)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
