"""Inspect preserved stage5 artifacts without changing generated pixels."""
from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'generation/cloud_results'
OUT = RESULTS / 'outputs'
DELIVERY = ROOT / 'delivery/stage5'


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    job = json.loads((RESULTS / 'job.json').read_text())
    roi = job['roi']; checks = []; measurements = []
    for name in job['primary_cases']:
        target = DELIVERY / name; target.mkdir(exist_ok=True)
        for video in sorted((OUT / name).glob('*.mp4')):
            shutil.copy2(video, target / video.name)
            data = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_frames', '-show_streams',
                '-select_streams', 'v:0', '-of', 'json', str(video)]))
            stream = data['streams'][0]
            pts = np.array([float(f['best_effort_timestamp_time']) for f in data['frames']])
            factor = 5 if 'slow_5x' in video.name else 1
            expected = np.array(job['elapsed_times_s']) * factor
            assert len(pts) == 16 and np.max(np.abs(pts - expected)) < 2e-5
            assert abs(float(stream['duration']) - (expected[-1] + job['last_frame_duration_s'] * factor)) < 2e-5
            checks.append({'video':str(video.relative_to(RESULTS)), 'frames':len(pts),
                           'size':[stream['width'],stream['height']], 'duration_s':float(stream['duration']),
                           'max_pts_error_s':float(np.max(np.abs(pts - expected)))})
        patch_sheet = Image.new('RGB', (4*300,4*174), '#222222')
        draw = ImageDraw.Draw(patch_sheet)
        for i,row in enumerate(job['cases'][name]['rows']):
            raw = Image.open(OUT/name/'raw_frames'/f'{i:04d}.png').convert('RGB')
            composite = np.array(Image.open(OUT/name/'composite_frames'/f'{i:04d}.png').convert('RGB'))
            background = np.array(Image.open(OUT/'shared_background'/f'{i:04d}.png').convert('RGB'))
            outside = np.ones(background.shape[:2],dtype=bool)
            outside[roi['top']:roi['top']+144,roi['left']:roi['left']+144] = False
            assert np.array_equal(composite[outside],background[outside])
            center = (np.array(row['uv_px']) - [roi['left'],roi['top']]) * (384/144)
            x,y = center
            condition = Image.open(OUT/'conditions'/name/f'{i:03d}.png').convert('RGB')
            box=(round(x)-36,round(y)-36,round(x)+36,round(y)+36)
            a=condition.crop(box).resize((144,144));b=raw.crop(box).resize((144,144))
            px=(i%4)*300;py=(i//4)*174
            patch_sheet.paste(a,(px,py+26));patch_sheet.paste(b,(px+150,py+26))
            status='uncertain_fused_with_helmet' if name=='ff_base' and i>=12 else 'visible'
            draw.text((px+4,py+5),f'{i}: condition | raw / {status[:9]}',fill='white')
            measurements.append({'case':name,'frame':i,'visibility':status,
                'projected_uv_px':row['uv_px'],'generated_center_px':None,
                'note':'No reliable independent center assigned to fused ball; no ground truth invented.' if i>=12 and name=='ff_base' else
                       'Visually identifiable ball; exact center not manually annotated.'})
        patch_sheet.save(DELIVERY/f'{name}_ball_patches.jpg',quality=95)
    for name in ['ff_base','sl_variant']:
        for kind in ['raw_frames','composite_frames']:
            sheet=Image.new('RGB',(4*384,4*408),'#222222');draw=ImageDraw.Draw(sheet)
            for i,f in enumerate(sorted((OUT/name/kind).glob('*.png'))):
                im=Image.open(f).convert('RGB')
                if kind=='composite_frames': im=im.crop((556,237,700,381)).resize((384,384))
                x=(i%4)*384;y=(i//4)*408;sheet.paste(im,(x,y+24));draw.text((x+8,y+5),f'{name} {kind} {i}',fill='white')
            sheet.save(DELIVERY/f'{name}_{kind}_contact.jpg',quality=94)
    for speed in ['normal','slow_5x']:
        for kind in ['raw','composite']:
            a=DELIVERY/'ff_base'/f'{kind}_{speed}.mp4';b=DELIVERY/'sl_variant'/f'{kind}_{speed}.mp4'
            filter_='[0:v][1:v]hstack=inputs=2[v]' if kind=='raw' else '[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack=inputs=2[v]'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(a),'-i',str(b),
                '-filter_complex',filter_,'-map','[v]','-fps_mode','vfr','-enc_time_base','1:90000',
                '-c:v','libx264','-bf','0','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                str(DELIVERY/f'{kind}_comparison_{speed}.mp4')],check=True)
    validation={'status':'cloud_run_completed_with_visual_limitations','download_hash_files':277,
        'raw_frames_preserved':32,'composite_frames_checked':32,'same_condition_first_frame':bool(np.array_equal(
            np.array(Image.open(OUT/'conditions/ff_base/000.png')),np.array(Image.open(OUT/'conditions/sl_variant/000.png')))),
        'all_composite_pixels_outside_roi_unchanged':True,'video_checks':checks,
        'visual_review':{'raw_and_composite_all_frames_reviewed':True,'ff_clear_ball_frames':12,
            'ff_uncertain_fused_frames':[12,13,14,15],'sl_clear_ball_frames':16,
            'clearly_missing_ball_frames':[],'obvious_duplicate_ball_frames':[],
            'physical_occlusion_masks_used':False,'exact_generator_position_error_not_measured':True,
            'limits':'No P90 claim: exact generated centers are unannotated. FF late ball merges with helmet highlights.'}}
    (ROOT/'reports/stage5_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (ROOT/'reports/stage5_frame_review.json').write_text(json.dumps(measurements,indent=2)+'\n')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>阶段五视频检查</title>
<style>body{font:17px system-ui;margin:32px;background:#141414;color:#eee}video,img{max-width:100%}section{margin:32px 0}a{color:#8ad}</style>
<h1>阶段五：FF / SL，同一场景</h1><p>左 FF，右 SL。模型生成各16帧，真实时长约0.267秒。FF末4帧球影与头盔亮部融合，不能认作准确遮挡处理。密集条件可能被复制，不代表恢复了真实自转或物理规律。</p>
<h2>局部合成，5倍慢放</h2><video controls loop src="composite_comparison_slow_5x.mp4"></video>
<h2>原始神经输出，5倍慢放</h2><video controls loop src="raw_comparison_slow_5x.mp4"></video>
<h2>正常速度</h2><video controls loop src="composite_comparison_normal.mp4"></video>
<p>局部合成只使用神经输出像素，没有在输出上再覆盖程序球。原始输出保留人物与好球框细节变化，因此请与合成分别查看。</p>'''
    for name in job['primary_cases']:
        html+=f'<section><h2>{name} 全部原始帧</h2><img src="{name}_raw_frames_contact.jpg"><h2>条件与生成球体局部对照</h2><img src="{name}_ball_patches.jpg"></section>'
    (DELIVERY/'index.html').write_text(html+'</html>')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
