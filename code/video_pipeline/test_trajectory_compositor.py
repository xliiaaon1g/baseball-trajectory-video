"""Deterministic 2D ball compositor; no diffusion model or learned renderer.

Manual observations locate the old ball for removal. Existing TrackNet-derived
conditions control the new ball. Pixel checks validate compositing only.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data/tracknet_ff_sl/ff_to_sl_model_preview'
OUT = ROOT / 'data/tracknet_ff_sl/compositor_test'
CASES = ['original_ff', 'sl_shape_candidate']
W, H, F, FPS = 832, 480, 81, 16


def encode(folder, target):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS),
                    '-i', str(folder/'%03d.png'), '-c:v', 'libx264', '-crf', '16',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)


def ball(plate, xy, previous):
    # Procedural disk, subpixel antialiasing and mild directional blur.
    # Its radius and shutter are display assumptions, not recovered 3D physics.
    x, y = xy
    radius = 2.2
    pad = 9
    left, top = int(x)-pad, int(y)-pad
    yy, xx = np.mgrid[top:top+2*pad+1, left:left+2*pad+1]
    delta = np.clip(np.asarray(xy)-previous, -3, 3)*.3
    alpha = np.zeros_like(xx, dtype=float)
    for amount in np.linspace(-.5, .5, 5):
        distance = np.hypot(xx-x-amount*delta[0], yy-y-amount*delta[1])
        alpha += np.clip(radius+.5-distance, 0, 1)/5
    # Small shaded white ball; not a second ball or an overlaid path line.
    shade = np.clip(241-8*(yy-y)/radius-5*(xx-x)/radius, 218, 250)
    result = plate.copy()
    roi = plate[top:top+len(yy), left:left+len(xx[0])].astype(float)
    result[top:top+len(yy), left:left+len(xx[0])] = np.rint(
        roi*(1-alpha[..., None])+shade[..., None]*alpha[..., None]).astype(np.uint8)
    support = np.zeros((H, W), np.uint8)
    support[top:top+len(yy), left:left+len(xx[0])] = (alpha>0).astype(np.uint8)
    return result, support


def decode_check(path):
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if image.shape[:2] != (H, W):
            raise ValueError('Unexpected video dimensions')
        count += 1
    cap.release()
    if count != F:
        raise ValueError('Incomplete video: '+str(path))
    return count


def main():
    from ball_control_experiment import validate_inputs
    contract = validate_inputs(INPUT)
    OUT.mkdir(exist_ok=True)
    comparison = json.loads((INPUT/'comparison.json').read_text())
    start, end = comparison['source_window_s']
    rows = sorted([q for q in json.loads((ROOT/'data/tracknet_ff_sl/dataset.json').read_text())['rows']
                   if q['sample_id']=='FF_02'], key=lambda q:q['decoded_pts'])
    times = np.array([q['decoded_pts'] for q in rows])
    centers = np.array([[q['x'],q['y']] for q in rows])
    cap = cv2.VideoCapture(str(ROOT/rows[0]['source_video']))
    source, pts = [], []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        source.append(image)
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if pts[-1] > end+.04:
            break
    cap.release()
    pts = np.asarray(pts)
    indices = np.abs(pts[:, None]-np.linspace(start, end, F)[None]).argmin(0)
    tracks = {case:np.load(INPUT/case/'tracks.npy')[:, 6] for case in CASES}
    folders = {}
    for mode in ['frozen_scene', 'source_motion']:
        for case in CASES:
            folder = OUT/mode/case/'frames'
            folder.mkdir(parents=True, exist_ok=True)
            folders[mode, case] = folder
    for name in ['clean_plate_frames', 'removal_masks', 'source_frames', 'ball_masks']:
        (OUT/name).mkdir(exist_ok=True)
    clean, original, removals = [], [], []
    source_records = []
    for i, index in enumerate(indices):
        frame = source[index]
        observed = np.array([np.interp(pts[index], times, centers[:, k]) for k in range(2)])
        # Remove only a small neighborhood at the independently annotated old ball.
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(mask, tuple(np.rint(observed).astype(int)), 8, 255, -1)
        # Labels are hints: inspect nearby compact bright pixels to cover old
        # ball residuals. This is local source cleanup, not a validated detector.
        hx,hy=np.rint(observed).astype(int)
        patch=frame[hy-26:hy+27,hx-26:hx+27].astype(float)
        low=patch.min(2)
        bright=((low>145)&(patch.max(2)-low<90)).astype(np.uint8)
        _,_,stats,candidates=cv2.connectedComponentsWithStats(bright,8)
        candidates=[c+np.array([hx-26,hy-26]) for s,c in zip(stats[1:],candidates[1:])
                    if 18<=s[cv2.CC_STAT_AREA]<=140]
        refined=None
        if candidates:
            q=min(candidates,key=lambda c:np.linalg.norm(c-observed))
            if np.linalg.norm(q-observed)<=18:
                refined=q
                cv2.circle(mask,tuple(np.rint(q).astype(int)),9,255,-1)
        repaired = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        resize = lambda a: cv2.copyMakeBorder(cv2.resize(a, (832,468)),6,6,0,0,cv2.BORDER_CONSTANT)
        original.append(resize(frame))
        clean.append(resize(repaired))
        removals.append(resize(mask))
        source_records.append({'frame':i, 'source_pts_s':float(pts[index]),
                               'old_ball_source_xy':observed.tolist(),
                               'cleanup_bright_component_xy':refined.tolist() if refined is not None else None})
        cv2.imwrite(str(OUT/'source_frames'/f'{i:03d}.png'), original[-1])
        cv2.imwrite(str(OUT/'clean_plate_frames'/f'{i:03d}.png'), clean[-1])
        cv2.imwrite(str(OUT/'removal_masks'/f'{i:03d}.png'), removals[-1])
    checks = []
    for mode in ['frozen_scene', 'source_motion']:
        for case in CASES:
            for i, xy in enumerate(tracks[case]):
                plate = clean[0] if mode=='frozen_scene' else clean[i]
                image, support = ball(plate, xy, tracks[case][max(0,i-1)])
                changed = np.any(image != plate, axis=2)
                if np.any(changed & (support==0)):
                    raise ValueError('Renderer altered pixels outside ball support')
                # Verify the visible change exists near the intended center.
                delta = np.abs(image.astype(float)-plate.astype(float)).sum(2)
                yy, xx = np.mgrid[:H,:W]
                recovered = np.array([(delta*xx).sum(),(delta*yy).sum()])/delta.sum()
                error = float(np.linalg.norm(recovered-xy))
                if error>2.:
                    raise ValueError('Composite center displaced beyond 2px')
                checks.append({'mode':mode,'case':case,'frame':i,'residual_center_error_px':error})
                cv2.imwrite(str(folders[mode,case]/f'{i:03d}.png'), image)
                if mode=='source_motion':
                    maskdir=OUT/'ball_masks'/case
                    maskdir.mkdir(exist_ok=True)
                    cv2.imwrite(str(maskdir/f'{i:03d}.png'),support*255)
            encode(folders[mode,case],OUT/mode/case/'composited.mp4')
        a,b=[cv2.imread(str(folders[mode,c]/'000.png')) for c in CASES]
        if not np.array_equal(a,b):
            raise ValueError('Paired first frames differ')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(OUT/mode/CASES[0]/'composited.mp4'),
                        '-i',str(OUT/mode/CASES[1]/'composited.mp4'),'-filter_complex','hstack=inputs=2',
                        '-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',
                        str(OUT/mode/'comparison.mp4')],check=True)
    encode(OUT/'clean_plate_frames', OUT/'clean_plate.mp4')
    # Enlarged audit: source / removed / FF / alternative, all at matching times.
    tiles=[]
    for i in [0,16,32,48,64,80]:
        panels=[original[i],clean[i]]+[cv2.imread(str(folders['source_motion',c]/f'{i:03d}.png')) for c in CASES]
        crops=[]
        for panel,label in zip(panels,['Source','Ball removed','FF composite','SL shape composite']):
            crop=cv2.resize(panel[164:252,360:468],(324,264),interpolation=cv2.INTER_NEAREST)
            tile=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(tile,f'{label} / {i}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            crops.append(tile)
        tiles.append(np.hstack(crops))
    cv2.imwrite(str(OUT/'audit_contact_sheet.jpg'),np.vstack(tiles))
    videos={}
    for mode in ['frozen_scene','source_motion']:
        for case in CASES:
            p=OUT/mode/case/'composited.mp4'
            videos[str(p.relative_to(OUT))]={'decoded_frames':decode_check(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    report={'status':'local_compositing_passed_visual_review_required','input_contract':contract,
            'output_size':[W,H],'frames':F,'fps':FPS,'source_records':source_records,
            'max_residual_center_error_px':max(q['residual_center_error_px'] for q in checks),
            'same_first_frame_each_pair':True,'outside_ball_support_changes':0,'videos':videos,
            'limitations':['Procedural compositing, no neural generation.',
                           'Old-ball removal uses manual hints plus local bright-component refinement, distinct from target paths.',
                           'Inpainting may affect local texture; inspect audit contact sheet.',
                           '2.2px display radius and blur are assumptions, not recovered physics.',
                           'Only retained visible flight segment; no release/catch or occlusion claim.',
                           'SL-shaped path is a 2D candidate, not a validated slider.',
                           'Residual-center metric is a compositor check, not a generated-domain detector.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    (OUT/'pixel_checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>二维球路合成测试</title>
<style>body{background:#111827;color:#eee;font:17px system-ui;max-width:1200px;margin:30px auto;padding:20px}video,img{width:100%}p{line-height:1.7}a{color:#93c5fd}</style>
<h1>同场景、两条二维球路：本地合成测试</h1><p>左：TrackNet FF 球路；右：SL 形状候选。两边共享首帧和背景。以下是确定性合成，未使用视频生成模型。球显示直径约 4.4 像素，慢放约 5 秒。</p>
<h2>保留原视频人物运动</h2><video controls loop src="source_motion/comparison.mp4"></video>
<p>先按人工标注移除原球，再沿目标轨迹加入一颗球。仅使用已审查的可见飞行片段，未模拟完整接球。</p>
<h2>仅用同一首帧的最小闭环</h2><video controls loop src="frozen_scene/comparison.mp4"></video>
<p>这一版人物与背景静止，用于隔离检验坐标、时间、渲染和导出。</p>
<h2>局部清除与合成审查</h2><img src="audit_contact_sheet.jpg"><p><a href="report.json">输入、校验和局限</a></p>''')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_records','input_contract']},indent=2))


if __name__=='__main__':
    main()
