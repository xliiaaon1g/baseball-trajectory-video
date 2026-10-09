"""Portable media utilities: no model loading and no hidden ball replacement."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def encode(directory,times,last_duration,out):
    times=np.asarray(times);micros=np.rint(times*1e6).astype(int)
    durations=list(np.diff(micros)/1e6)+[last_duration]
    text=['ffconcat version 1.0']
    for i,t in enumerate(durations):text += [f"file '{i:04d}.png'",'option framerate 90000',f'duration {t:.6f}']
    text += [f"file '{len(times)-1:04d}.png'",'option framerate 90000']
    concat=directory/'frames.ffconcat';concat.write_text('\n'.join(text)+'\n')
    subprocess.run([shutil.which('ffmpeg') or 'ffmpeg','-nostdin','-v','error','-y','-safe','0','-f','concat','-i',str(concat),
        '-frames:v',str(len(times)),'-fps_mode','vfr','-enc_time_base','1:90000','-c:v','libx264','-bf','0','-crf','18',
        '-pix_fmt','yuv420p','-bsf:v',f'setts=duration={round(last_duration*90000)}:time_base=1/90000',
        '-video_track_timescale','90000','-movflags','+faststart',str(out)],check=True)


def composite_sprite(bg,xy,rgba,scale,angle):
    rgba=cv2.resize(rgba,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    alpha=rgba[:,:,3].astype(np.float32)/255;premul=rgba[:,:,:3].astype(np.float32)*alpha[:,:,None]
    yy,xx=np.indices(alpha.shape);center=np.array([(xx*alpha).sum(),(yy*alpha).sum()])/alpha.sum()
    rotation=cv2.getRotationMatrix2D(tuple(center),-angle,1)
    alpha=cv2.warpAffine(alpha,rotation,(alpha.shape[1],alpha.shape[0]));premul=cv2.warpAffine(premul,rotation,(alpha.shape[1],alpha.shape[0]))
    M=np.float32([[1,0,xy[0]-center[0]],[0,1,xy[1]-center[1]]]);h,w=bg.shape[:2]
    aa=cv2.warpAffine(alpha,M,(w,h));pp=cv2.warpAffine(premul,M,(w,h))
    result=np.rint(np.clip(bg*(1-aa[:,:,None])+pp,0,255)).astype(np.uint8)
    assert not np.any(np.any(result!=bg,axis=2)&(aa<=1e-5))
    return result,aa


def build_conditions(root,repair_frames):
    job=json.loads((root/'job.json').read_text());output=root/'outputs';output.mkdir(exist_ok=True)
    roi=job['roi'];crop=job['repair_crop'];backgrounds=[];audit=[]
    for i,idx in enumerate(job['target_context_indices']):
        source=cv2.imread(str(root/'source_frames'/f'{i:04d}.png'));repaired=cv2.imread(str(repair_frames/f'{idx:04d}.png'))
        assert repaired is not None and repaired.shape==(256,256,3)
        mask=cv2.imread(str(root/'context_masks'/f'{idx:04d}.png'),0)
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
        region=source[crop['top']:crop['top']+256,crop['left']:crop['left']+256]
        bg=source.copy();bg[crop['top']:crop['top']+256,crop['left']:crop['left']+256]=np.where(mask[:,:,None]>0,repaired,region)
        directory=output/'shared_background';directory.mkdir(exist_ok=True);cv2.imwrite(str(directory/f'{i:04d}.png'),bg);backgrounds.append(bg)
        if i in [0,5,10,15]:
            audit.append(np.hstack([source[crop['top']:crop['top']+256,crop['left']:crop['left']+256],bg[crop['top']:crop['top']+256,crop['left']:crop['left']+256]]))
    cv2.imwrite(str(output/'repair_audit.png'),np.vstack(audit))
    bank=[cv2.imread(str(root/'sprites'/f'{i:03d}.png'),cv2.IMREAD_UNCHANGED) for i in range(5)]
    positions=np.linspace(0,4,16);first_images=[];condition_audit=[]
    for name in job['primary_cases']:
        directory=output/'conditions'/name;directory.mkdir(parents=True,exist_ok=True)
        support_dir=output/'supports'/name;support_dir.mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(job['cases'][name]['rows']):
            lo=int(np.floor(positions[i]));hi=min(4,lo+1);fraction=positions[i]-lo
            a=(1-fraction)*bank[lo][:,:,3].astype(float)/255+fraction*bank[hi][:,:,3].astype(float)/255
            pp=(1-fraction)*bank[lo][:,:,:3]*(bank[lo][:,:,3,None]/255.)+fraction*bank[hi][:,:,:3]*(bank[hi][:,:,3,None]/255.)
            rgb=np.divide(pp,np.maximum(a[:,:,None],.001));sprite=np.dstack([np.rint(rgb).clip(0,255),np.rint(a*255)]).astype(np.uint8)
            active=a>.25;yy,xx=np.indices(a.shape);diameter=max(np.ptp(xx[active])+1,np.ptp(yy[active])+1)
            source_direction=job['cases']['ff_base']['rows'][0]['screen_direction_deg']
            angle=0 if i==0 else r['screen_direction_deg']-source_direction
            frame,alpha=composite_sprite(backgrounds[i],r['uv_px'],sprite,r['nominal_diameter_px']/diameter,angle)
            condition=cv2.resize(frame[roi['top']:roi['top']+roi['height'],roi['left']:roi['left']+roi['width']],(384,384),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(directory/f'{i:03d}.png'),condition)
            cv2.imwrite(str(support_dir/f'{i:04d}.png'),np.rint(alpha*255).astype(np.uint8))
            if i==0:first_images.append(condition)
            if i in [0,5,10,15]:condition_audit.append(condition)
    assert np.array_equal(first_images[0],first_images[1])
    cv2.imwrite(str(output/'condition_audit.png'),np.vstack([np.hstack(condition_audit[:4]),np.hstack(condition_audit[4:])]))
    dump(output/'conditions_manifest.json',{'status':'prepared_repaired_background_visual_review_required','same_first_frame':True,
         'repair_source':str(repair_frames),'bank_frames':5,'empirical_blur_rotated':True,'physical_spin_recovered':False,
         'cases':job['primary_cases'],'frames':16,'size':[384,384]})


def export_outputs(root,name,frames):
    job=json.loads((root/'job.json').read_text());out=root/'outputs'/name
    raw=out/'raw_frames';full=out/'composite_frames';masks=out/'blend_masks'
    for d in [raw,full,masks]:d.mkdir(parents=True,exist_ok=True)
    roi=job['roi'];cx,cy,w,h=roi['left'],roi['top'],roi['width'],roi['height']
    for i,frame in enumerate(frames):
        assert frame.size==(384,384);frame.save(raw/f'{i:04d}.png')
        # Blend actual neural pixels. Never add a programmatic ball afterward.
        bg=cv2.imread(str(root/'outputs/shared_background'/f'{i:04d}.png'))
        support=cv2.imread(str(root/'outputs/supports'/name/f'{i:04d}.png'),0)
        mask=cv2.dilate((support>0).astype(np.uint8),np.ones((7,7),np.uint8)).astype(float)
        mask=cv2.GaussianBlur(mask,(9,9),1.8);mask=np.clip(mask,0,1)
        generated=cv2.resize(cv2.cvtColor(np.array(frame),cv2.COLOR_RGB2BGR),(w,h),interpolation=cv2.INTER_CUBIC)
        region=bg[cy:cy+h,cx:cx+w].astype(float);alpha=mask[cy:cy+h,cx:cx+w,None]
        result=bg.copy();result[cy:cy+h,cx:cx+w]=np.rint(region*(1-alpha)+generated*alpha).clip(0,255).astype(np.uint8)
        assert not np.any(np.any(result!=bg,axis=2)&(mask==0))
        cv2.imwrite(str(full/f'{i:04d}.png'),result);cv2.imwrite(str(masks/f'{i:04d}.png'),np.rint(mask*255).astype(np.uint8))
    times=np.array(job['elapsed_times_s']);last=job['last_frame_duration_s']
    for folder,label in [(raw,'raw'),(full,'composite')]:
        for factor,speed in [(1,'normal'),(5,'slow_5x')]:encode(folder,times*factor,last*factor,out/f'{label}_{speed}.mp4')
