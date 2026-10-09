"""Local ball-control contract, synthetic closed loop, blind export and scoring.

Synthetic detection is a fixture-only pixel reader, not a broadcast detector.
Generated images require independent annotations; reference paths are only
joined AFTER annotation. No generative weights are loaded by the local run.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'data/tracknet_ff_sl/ff_to_sl_model_preview'
DEFAULT_OUT = ROOT / 'data/tracknet_ff_sl/local_framework'
CASES = ['original_ff', 'sl_shape_candidate']
W, H, F, FPS = 832, 480, 81, 16


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(inputs):
    arrays, hashes, metadata = [], [], []
    for name in CASES:
        folder = inputs / name
        tracks = np.load(folder / 'tracks.npy', allow_pickle=False)
        visibility = np.load(folder / 'visibility.npy', allow_pickle=False)
        meta = json.loads((folder / 'metadata.json').read_text())
        if tracks.shape != (F, 7, 2) or tracks.dtype != np.float32 or not np.isfinite(tracks).all():
            raise ValueError('Invalid track shape/dtype/values: ' + name)
        if visibility.shape != (F, 7) or visibility.dtype != np.bool_ or not visibility.all():
            raise ValueError('The experiment needs boolean, all-visible conditions: ' + name)
        if not ((tracks[...,0] >= 0).all() and (tracks[...,0] < W).all()
                and (tracks[...,1] >= 0).all() and (tracks[...,1] < H).all()):
            raise ValueError('Coordinates outside image: ' + name)
        with Image.open(folder / 'first_frame.png') as first:
            if first.size != (W,H):
                raise ValueError('First-frame dimensions: ' + name)
        if not all(meta.get(k) == v for k,v in {'frame_count':F,'width':W,'height':H,'fps':FPS,'ball_index':6}.items()):
            raise ValueError('Metadata contract mismatch: ' + name)
        if not np.allclose(tracks[:,:6], tracks[0,:6][None]):
            raise ValueError('Background anchors move: ' + name)
        arrays.append(tracks)
        hashes.append(sha(folder / 'first_frame.png'))
        metadata.append(meta)
    if len(set(hashes)) != 1:
        raise ValueError('Paired first frames differ')
    if not np.allclose(arrays[0][:,:6], arrays[1][:,:6]):
        raise ValueError('Paired anchors differ')
    if not np.allclose(arrays[0][[0,-1],6], arrays[1][[0,-1],6]):
        raise ValueError('Paired endpoints differ')
    sampled = np.arange(0,F,4)
    a,b = [np.floor(tracks[sampled,6]/8).astype(int) for tracks in arrays]
    changed = int(np.any(a != b,axis=1).sum())
    if not changed:
        raise ValueError('Conditions become identical on the pinned model grid')
    return {'status':'input_contract_passed','shape':[F,7,2], 'first_frame_sha256':hashes[0],
            'changed_ball_grid_samples':changed,'total_grid_samples':len(sampled),
            'condition_sha256':{name:sha(inputs/name/'tracks.npy') for name in CASES},
            'scope':'retained 2D flight segment; not verified full release-to-glove or physical SL',
            'source_duration_s':metadata[0]['flight_duration_source_s']}


def components(mask):
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [(centers[i],int(stats[i,cv2.CC_STAT_AREA])) for i in range(1,count)
            if stats[i,cv2.CC_STAT_AREA] >= 3]


def detect_fixture(image):
    # All-image search; no expected point, reference, ROI or target size.
    found = components((image > 220).all(2))
    status = 'absent' if not found else 'unique' if len(found)==1 else 'multiple'
    return {'status':status,'center_px':found[0][0].tolist() if status=='unique' else None,
            'diameter_px':2*math.sqrt(found[0][1]/math.pi) if status=='unique' else None}


def scene_points(image):
    # Teal corner markers exist ONLY in the synthetic calibration scene.
    return np.array([c for c,_ in components((image[:,:,0]<80)&(image[:,:,1]>130)&(image[:,:,2]>150))])


def score_rows(rows, tracks, scene_ok=None):
    if (len(rows) != F or any(type(r.get('frame')) is not int for r in rows)
            or {r['frame'] for r in rows} != set(range(F))):
        raise ValueError('Need exactly frames 0..80 without duplicates')
    rows = sorted(rows,key=lambda r:r['frame'])
    result, errors = [], []
    for row in rows[1:]:
        frame, status = row['frame'], row['status']
        if status not in {'unique','absent','multiple','uncertain'}:
            raise ValueError('Unreviewed/invalid scored frame: ' + str(frame))
        error = None
        if status == 'unique':
            center = row.get('center_px')
            if not isinstance(center,list) or len(center)!=2 or not all(isinstance(v,(int,float))
                      and not isinstance(v,bool) and math.isfinite(v) for v in center):
                raise ValueError('Unique ball needs finite center: ' + str(frame))
            if not (0 <= center[0] < W and 0 <= center[1] < H):
                raise ValueError('Center outside image')
            error = float(np.linalg.norm(np.array(center)-tracks[frame,6]))
            errors.append(error)
        passed = error is not None and error <= 5.
        result.append({'frame':frame,'status':status,'error_px':error,'path_success':passed,
                       'scene_stable':bool(scene_ok[frame]) if scene_ok is not None else None})
    success = sum(r['path_success'] for r in result)/80
    strict = (sum(r['path_success'] and r['scene_stable'] for r in result)/80
              if scene_ok is not None else None)
    return {'scored_frames':80,'excluded_first_frame':True,'tolerance_px':5.,
            'path_success_fraction':success,'path_and_scene_success_fraction':strict,
            'unique_frames':sum(r['status']=='unique' for r in result),
            'status_counts':{s:sum(r['status']==s for r in result) for s in ['unique','absent','multiple','uncertain']},
            'median_error_px_on_unique_frames':float(np.median(errors)) if errors else None,
            'p90_error_px_on_unique_frames':float(np.percentile(errors,90)) if errors else None,
            'scene_review':'synthetic pixel check' if scene_ok is not None else 'independent review pending',
            'frames':result}


def encode(frames, target, lossless=False):
    codec = ['-c:v','ffv1'] if lossless else ['-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart']
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(FPS),'-i',str(frames/'%03d.png'),
                    *codec,str(target)],check=True)


def export_blind(outputs, records, destination):
    if destination.exists():
        raise ValueError('Refuse overwrite of blind review data: '+str(destination))
    destination.mkdir(parents=True)
    order = list(records)
    random.Random(31).shuffle(order)
    mapping = []
    for i, record in enumerate(order,1):
        blind_id = f'clip_{i:03d}'
        folder = destination/blind_id
        folder.mkdir()
        frames = outputs/record['output_case']/'frames'
        files = sorted(frames.glob('*.png'))
        if [f.name for f in files] != [f'{i:03d}.png' for i in range(F)]:
            raise ValueError('Invalid output frame sequence: '+record['output_case'])
        for path in files:
            with Image.open(path) as frame:
                if frame.size != (W,H):
                    raise ValueError('Output dimensions mismatch')
            shutil.copyfile(path,folder/path.name)
        mapping.append({'blind_id':blind_id,**record})
    dump(destination.parent/'blind_mapping_KEEP_FROM_ANNOTATOR.json',mapping)
    shutil.copyfile(ROOT.parent/'pilot_pipeline/annotation.html',destination.parent/'annotation.html')
    return mapping


def evaluate_annotation(payload, mapping, inputs, allow_fixture=False):
    if payload.get('schema') != 'blind_ball_annotations_v1' or (payload.get('width'),payload.get('height')) != (W,H):
        raise ValueError('Annotation schema/dimensions mismatch')
    if payload.get('method')=='synthetic fixture detector' and not allow_fixture:
        raise ValueError('Synthetic detector annotations cannot establish generated-video performance')
    records = [r for r in mapping if r['blind_id']==payload.get('blind_id')]
    if len(records)!=1:
        raise ValueError('Unknown/duplicate blind ID')
    record = records[0]
    if record['condition'] not in CASES:
        raise ValueError('Unknown condition')
    tracks = np.load(inputs/record['condition']/'tracks.npy',allow_pickle=False)
    if record.get('condition_sha256') != sha(inputs/record['condition']/'tracks.npy'):
        raise ValueError('Conditions changed after output/annotation export')
    score = score_rows(payload['annotations'],tracks)
    score.update(blind_id=payload['blind_id'],condition=record['condition'],
                 evaluation_kind='synthetic round-trip' if allow_fixture else 'independent annotations',
                 condition_sha256=sha(inputs/record['condition']/'tracks.npy'))
    return score


def local_run(inputs,out):
    contract = validate_inputs(inputs)
    if out.exists():
        raise ValueError('Refuse overwrite of experiment: '+str(out))
    out.mkdir(parents=True)
    dump(out/'input_contract.json',contract)
    fixtures = [('reference_ff','original_ff',None),('reference_sl','sl_shape_candidate',None),
                ('missing','original_ff','missing'),('shift16','original_ff','shift'),
                ('multiple','original_ff','multiple'),('background_shift','original_ff','background')]
    records, reports = [], {}
    for name,condition,fault in fixtures:
        tracks = np.load(inputs/condition/'tracks.npy',allow_pickle=False)
        folder = out/name
        frames = folder/'frames'
        frames.mkdir(parents=True)
        rows,stable,initial = [],[],None
        for frame in range(F):
            image = Image.new('RGB',(W,H),(24,32,43))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,320,W,H),fill=(26,66,42))
            for x,y in tracks[frame,:6]:
                x,y = round(float(x)),round(float(y))
                draw.rectangle((x-2,y-2,x+2,y+2),fill=(35,165,185))
            x,y = tracks[frame,6].astype(float)
            if frame and fault=='shift':
                y += 16
            if not (frame and fault in {'missing','background'}):
                x,y=round(x),round(y)
                draw.ellipse((x-3,y-3,x+3,y+3),fill=(250,250,250))
            if frame and fault=='multiple':
                draw.ellipse((300,397,306,403),fill=(250,250,250))
            rgb = np.array(image)
            if frame and fault=='background':
                rgb = cv2.warpAffine(rgb,np.float32([[1,0,8],[0,1,0]]),(W,H),borderValue=(24,32,43))
                # Move the scene while keeping the ball on its correct path:
                # path-only scoring should pass, scene-plus-path should fail.
                shifted = Image.fromarray(rgb)
                overlay = ImageDraw.Draw(shifted)
                x,y=round(float(tracks[frame,6,0])),round(float(tracks[frame,6,1]))
                overlay.ellipse((x-3,y-3,x+3,y+3),fill=(250,250,250))
                rgb = np.array(shifted)
            Image.fromarray(rgb).save(frames/f'{frame:03d}.png')
            detected = detect_fixture(rgb)
            rows.append({'frame':frame,**detected})
            corners = scene_points(rgb)
            if initial is None:
                initial = corners
            drift = max(min(np.linalg.norm(corners-c,axis=1)) for c in initial) if len(corners)==len(initial) else float('inf')
            stable.append(drift <= 2.)
        score = score_rows(rows,tracks,stable)
        score['evaluation_kind']='synthetic software calibration; not generative inference'
        dump(folder/'metrics.json',score)
        dump(folder/'fixture_annotations.json',{'schema':'blind_ball_annotations_v1','width':W,'height':H,
             'method':'synthetic fixture detector','annotations':rows})
        encode(frames,folder/'preview.mp4')
        encode(frames,folder/'lossless.mkv',True)
        reports[name]={k:v for k,v in score.items() if k!='frames'}
        records.append({'output_case':name,'condition':condition,'backend':'synthetic_fixture',
                        'condition_sha256':sha(inputs/condition/'tracks.npy')})
    assert reports['reference_ff']['path_success_fraction']==1
    assert reports['reference_sl']['path_success_fraction']==1
    for name in ['missing','shift16','multiple']:
        assert reports[name]['path_success_fraction']==0, name
    assert reports['background_shift']['path_success_fraction']==1
    assert reports['background_shift']['path_and_scene_success_fraction']==0
    mapping = export_blind(out,records,out/'blind_frames')
    for record in mapping:
        payload=json.loads((out/record['output_case']/'fixture_annotations.json').read_text())
        payload['blind_id']=record['blind_id']
        result=evaluate_annotation(payload,mapping,inputs,allow_fixture=True)
        assert result['path_success_fraction']==reports[record['output_case']]['path_success_fraction']
    # Cross-condition test: the SL render must not pass as an FF path in all frames.
    cross=json.loads((out/'reference_sl/fixture_annotations.json').read_text())['annotations']
    wrong=score_rows(cross,np.load(inputs/'original_ff/tracks.npy'))
    assert wrong['path_success_fraction']<1
    # Verify one entire FFV1 stream round-trips exactly to PNG pixels.
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(out/'reference_ff/lossless.mkv'),
                                 '-f','rawvideo','-pix_fmt','rgb24','-'])
    expected=b''.join(np.array(Image.open(out/'reference_ff/frames'/f'{i:03d}.png')).tobytes() for i in range(F))
    assert raw==expected
    report={'status':'local_framework_passed','backend':'synthetic fixture, no generative weights',
            'input_contract':contract,'cases':reports,'wrong_condition_success':wrong['path_success_fraction'],
            'blind_export_and_score_roundtrip':'passed','ffv1_exact_frames_verified':F,
            'autodl_generation_status':'not executed','full_release_window_status':'still needs review'}
    dump(out/'report.json',report)
    body='''<!doctype html><meta charset="utf-8"><title>球路控制本地验收</title>
<style>body{font:17px system-ui;background:#101827;color:#eee;max-width:1050px;margin:30px auto;padding:20px}p{line-height:1.7}video{width:100%;max-width:832px}td,th{padding:10px;text-align:left}a{color:#8cd5ff}</style>
<h1>球路控制实验：本地框架已跑通</h1><p>这里使用可控合成画面校验数据接口、输出、盲标与评分。没有加载生成模型，不能据此判断 Wan-Move 的生成效果。
FF／SL 输入来自当前保留的二维球路片段，出手初段待复核，SL 候选尚未物理校准。</p>
<table><tr><th>样例</th><th>路径成功比例（≤5px 且唯一）</th><th>路径＋背景成功比例</th></tr>'''
    for name,score in reports.items():
        body+=f'<tr><td>{name}</td><td>{score["path_success_fraction"]:.1%}</td><td>{score["path_and_scene_success_fraction"]:.1%}</td></tr>'
    body+='</table><p>首帧不评分；其余 80 帧均计入，包括缺球、重复球和不确定状态。背景检查仅适用于这些合成样例。</p>'
    for name in ['reference_ff','reference_sl','missing','shift16','multiple','background_shift']:
        body+=f'<h2>{name}</h2><video controls loop src="{name}/preview.mp4"></video>'
    body+='<p><a href="annotation.html">独立盲标工具</a> · <a href="report.json">完整验收记录</a></p>'
    (out/'index.html').write_text(body)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['check','local','blind','evaluate']:
        q=sub.add_parser(command)
        q.add_argument('--inputs',type=Path,default=DEFAULT_INPUT)
        if command=='local':q.add_argument('--out',type=Path,default=DEFAULT_OUT)
        if command=='blind':
            q.add_argument('--outputs',type=Path,required=True)
            q.add_argument('--out',type=Path,required=True)
        if command=='evaluate':
            q.add_argument('--annotations',type=Path,required=True)
            q.add_argument('--mapping',type=Path,required=True)
            q.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='check':print(json.dumps(validate_inputs(args.inputs),indent=2))
    elif args.command=='local':local_run(args.inputs,args.out)
    elif args.command=='blind':
        validate_inputs(args.inputs)
        records=[]
        for name in CASES:
            run=json.loads((args.outputs/name/'run.json').read_text())
            if run.get('status')!='generated_manual_review_pending':
                raise ValueError('Generation not complete: '+name)
            if run.get('tracks_sha256')!=sha(args.inputs/name/'tracks.npy'):
                raise ValueError('Generation used different conditions: '+name)
            records.append({'output_case':name,'condition':name,'backend':'Wan-Move',
                            'condition_sha256':run['tracks_sha256']})
        export_blind(args.outputs,records,args.out)
    else:
        validate_inputs(args.inputs)
        payload=json.loads(args.annotations.read_text())
        mapping=json.loads(args.mapping.read_text())
        score=evaluate_annotation(payload,mapping,args.inputs)
        dump(args.out,score)
        print(json.dumps({k:v for k,v in score.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
