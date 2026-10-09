"""Verify delivered pair background invariance, first frame, and video decoding."""
from pathlib import Path
import hashlib,json,subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent/'data/tracknet_ff_sl/appearance_transfer'
report=json.loads((ROOT/'report.json').read_text());n=report['frames'];checks=[]
for variant in ['observed_appearance','exposure_blur']:
    for i in range(n):
        plate=cv2.imread(str(ROOT/'clean_plates'/f'{i:03d}.png'))
        a,b=[cv2.imread(str(ROOT/variant/c/'frames'/f'{i:03d}.png')) for c in ['original_ff','sl_shape_candidate']]
        masks=[cv2.imread(str(ROOT/variant/c/'alpha_support'/f'{i:03d}.png'),0) for c in ['original_ff','sl_shape_candidate']]
        for image,mask in zip([a,b],masks):
            assert not np.any(np.any(image!=plate,axis=2)&(mask==0)), 'Compositor modified common plate'
        outside=int(np.count_nonzero(np.any(a!=b,axis=2)&((masks[0]|masks[1])==0)))
        assert outside==0
        if i==0:assert np.array_equal(a,b), 'Different first frame'
        checks.append({'variant':variant,'frame':i,'pair_difference_outside_ball_support':outside,'pair_identical':bool(np.array_equal(a,b))})
video_checks={}
for path in ROOT.rglob('*.mp4'):
    q=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate:format=duration','-of','json',str(path)],capture_output=True,text=True,check=True).stdout)
    assert int(q['streams'][0]['nb_read_frames'])==n
    q['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();video_checks[str(path.relative_to(ROOT))]=q
(ROOT/'verification.json').write_text(json.dumps({'frame_pair_checks':checks,'videos':video_checks},indent=2))
print(f'Passed: {len(checks)} frame pairs; {len(video_checks)} complete videos; common plate outside ball support; identical first frames.')
