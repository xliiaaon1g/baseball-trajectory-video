"""Build a portable cloud job; no cloud resource or model download is started."""
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from export_stage1 import dump, probe, sha256

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[1]


def main():
    out=EXP/'generation';out.mkdir(exist_ok=True)
    package=out/'stage5_job';package.mkdir(exist_ok=True)
    data=json.loads((EXP/'training/prepared_data.json').read_text())
    edit=json.loads((EXP/'edits/stage4_manifest.json').read_text())
    sample=next(s for s in data['samples'] if s['sample_id']=='FF_02')
    source=Path(sample['source_video']);assert sha256(source)==sample['source_sha256']
    # Half a second of context on both sides, with all reviewed flight masks.
    count=len(probe(source)['frames']);first=max(0,edit['source_frame_ids'][0]-30)
    last=min(count-1,edit['source_frame_ids'][-1]+30)
    crop={'left':500,'top':181,'width':256,'height':256}
    for name in ['context_frames','context_masks','source_frames','sprites']:(package/name).mkdir(exist_ok=True)
    observations={r['frame_index']:r for r in sample['rows'] if r['use_for_observation_loss']}
    frame_metadata=[];cap=cv2.VideoCapture(str(source));i=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if first<=i<=last:
            cropped=image[crop['top']:crop['top']+256,crop['left']:crop['left']+256]
            mask=np.zeros((256,256),np.uint8)
            row=observations.get(i)
            if row:cv2.circle(mask,(round(row['u_px'])-crop['left'],round(row['v_px'])-crop['top']),12,255,-1)
            cv2.imwrite(str(package/'context_frames'/f'{i-first:04d}.png'),cropped)
            cv2.imwrite(str(package/'context_masks'/f'{i-first:04d}.png'),mask)
            frame_metadata.append({'decoded_frame_index':i,'context_index':i-first,'has_old_ball_mask':bool(row)})
            if i in edit['source_frame_ids']:
                index=edit['source_frame_ids'].index(i)
                cv2.imwrite(str(package/'source_frames'/f'{index:04d}.png'),image)
        i+=1
    cap.release();assert i==count
    for i in range(5):shutil.copy2(ROOT/'data/tracknet_ff_sl/appearance_transfer/sprites'/f'{i:03d}.png',package/'sprites'/f'{i:03d}.png')
    env=json.loads((ROOT.parent/'runpod_pilot/small_model_test/cloud_results/trial_004/environment.json').read_text())
    cases={c['case_id']:c for c in edit['cases'] if c['case_id'] in edit['primary_cases']}
    job={'schema_version':1,'stage':5,'status':'prepared_awaiting_cloud_budget','cases':cases,'primary_cases':edit['primary_cases'],
         'roi':edit['roi'],'repair_crop':crop,'context_frames':frame_metadata,
         'target_context_indices':[i-first for i in edit['source_frame_ids']],
         'source_video_pts_s':edit['source_pts_s'],'elapsed_times_s':edit['elapsed_times_s'],
         'last_frame_duration_s':edit['last_frame_duration_s'],'duration_s':edit['duration_s'],
         'models':env['models'],'revisions':env['revisions'],
         'steps':20,'seed':101,'guidance_scale':6.,'control_scale':1.,'image_size':[384,384],
         'repair':{'implementation':'official sczhou/ProPainter','crop_size':[256,256],'mask_radius_px':12,'mask_dilation_px':4,
                   'neighbor_length':10,'ref_stride':5,'subvideo_length':80,'fp16':True,
                   'must_visually_review_before_generation':True},
         'appearance':{'bank_count':5,'method':'smooth premultiplied-alpha blend of reviewed real sprites; rotate empirical blur with screen direction',
                       'size':'f * nominal diameter / predicted camera depth; stage4 preserved',
                       'new_exposure_blur':False,'real_spin_inferred':False},
         'prompt':'Realistic baseball broadcast footage, a catcher and umpire moving naturally, one small white baseball following the supplied frames, natural motion blur, fixed camera.',
         'negative_prompt':'multiple baseballs, duplicate ball, distorted face, text artifacts, cartoon',
         'compositing':'neural output pasted only within softened predicted ball support onto shared repaired background; raw frames retained and evaluated separately',
         'first_frame_policy':'retain actual neural first frames; no exact-preservation claim',
         'source_sha256':sample['source_sha256'],'model_ode_sha256':edit['model_sha256'],
         'cloud_plan':{'provider':'RunPod','reuse_pod_id':'vyc68xqn7lqv01','gpu':'A40 48GB','observed_restart_rate_usd_hour':.49,
                       'observed_balance_usd':14.75,'proposed_budget_usd':2.,'planned_max_session_minutes':30,
                       'stop_policy':'stop pod after results downloaded and verified, or at session limit; no new volume',
                       'budget_authorized':False,'cache_retention_unverified':True},
         'scope_limit':'reference-state-derived edited paths, same local effective camera; not independent physical counterfactual truth'}
    # Strip local data-path fields: the bundle is independently portable.
    for c in cases.values():
        for row in c['rows']:
            for k in ['rgb_condition_path','source_sprite_path','alpha_support_path']:row.pop(k,None)
    dump(package/'job.json',job)
    for name in ['run_stage5.py','stage5_media.py']:shutil.copy2(EXP/name,package/name)
    (package/'requirements.txt').write_text('diffusers==0.39.0\ntransformers==4.57.6\naccelerate==1.15.0\nhuggingface_hub==0.36.2\npeft==0.21.2\nsafetensors==0.8.0\nPillow\nopencv-python-headless\nscipy\naddict\neinops\nfuture\nscikit-image\nimageio-ffmpeg\npyyaml\nrequests\ntimm\nyapf\nmatplotlib\nav\n')
    (package/'README.md').write_text('# 阶段五运行包\n\n当前只准备了数据、配置和脚本，未启动计费实例。\n\n先复用现有 CUDA PyTorch 环境，不替换 torch/torchvision。按需安装 requirements.txt。克隆官方 sczhou/ProPainter，记录 git commit；额外模型按官方代码下载。\n\n```sh\npython run_stage5.py\npython run_stage5.py --repair --propainter /workspace/ProPainter\n```\n\n检查 outputs/repair_audit.png 和 condition_audit.png 后，再运行：\n\n```sh\npython run_stage5.py --generate --reviewed-repair\n```\n\n--repair 和 --generate 均要求 CUDA，会下载缺失模型。每条生成 16 帧，保留原始神经帧、软掩膜合成帧、正常速度和 5 倍慢放。不会用程序球替换神经输出。运行并下载结果后停止实例；新费用需先获得本轮预算授权。\n')
    files={str(f.relative_to(package)):sha256(f) for f in sorted(package.rglob('*')) if f.is_file()}
    dump(out/'stage5_package_validation.json',{'status':'portable_inputs_checked','context_frames':len(frame_metadata),
        'masked_context_frames':sum(f['has_old_ball_mask'] for f in frame_metadata),'target_frames':16,'cases':list(cases),
        'source_unchanged':sha256(source)==sample['source_sha256'],'neural_generator_run':False,'cloud_started':False,
        'runtime_cuda_required':True,'files':files})
    archive=out/'stage5_job.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(package.parent)))
    print(json.dumps({'archive':str(archive),'size_MB':archive.stat().st_size/1e6,'context_frames':len(frame_metadata),
                      'status':'prepared; cloud budget pending'},indent=2),flush=True)


if __name__=='__main__':main()
