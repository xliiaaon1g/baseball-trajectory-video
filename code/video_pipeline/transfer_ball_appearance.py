"""Real-video ball matte transfer baseline, not a trained appearance model.

Shared moving plates; fixed TrackNet/candidate routes; empirically extracted
RGBA sprites, with an explicitly assumed exposure integration variant.
"""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/tracknet_ff_sl'
OUT=DATA/'appearance_transfer'
CASES=['original_ff','sl_shape_candidate']

def encode(folder,target,fps):
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-framerate',str(fps),'-i',str(folder/'%03d.png'),'-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)

def component_center(frame,hint):
    x,y=np.rint(hint).astype(int)
    patch=frame[y-26:y+27,x-26:x+27].astype(float)
    low=patch.min(2)
    _,_,stats,centers=cv2.connectedComponentsWithStats(((low>145)&(patch.max(2)-low<90)).astype('uint8'),8)
    candidates=[c+[x-26,y-26] for s,c in zip(stats[1:],centers[1:]) if 18<=s[4]<=140]
    if not candidates:raise ValueError('No source candidate')
    return min(candidates,key=lambda p:np.linalg.norm(p-hint))

def sprite(frame,center):
    """Estimate a local alpha from contrast against an inpainted plate.
    Unmatting is approximate; do not call estimated alpha ground truth.
    """
    size=41;half=size//2
    patch=cv2.getRectSubPix(frame,(size,size),tuple(map(float,center))).astype(np.float32)
    yy,xx=np.mgrid[:size,:size];dist=np.hypot(xx-half,yy-half)
    bg=cv2.inpaint(patch.astype('uint8'),(dist<9).astype('uint8')*255,3,cv2.INPAINT_TELEA).astype(float)
    contrast=(patch-bg).mean(2)
    core=(dist<5)&(contrast>20)
    if core.sum()<8:raise ValueError('Not enough ball contrast')
    white=np.percentile(patch[core],90,axis=0)
    alpha=np.clip(contrast/np.maximum(white.mean()-bg.mean(2),15),0,1)
    alpha*=np.clip(8-dist,0,1)
    alpha[contrast<5]=0
    alpha=cv2.GaussianBlur(alpha.astype(np.float32),(3,3),.35)
    fg=np.clip((patch-bg*(1-alpha[...,None]))/np.maximum(alpha[...,None],.02),0,255)
    premul=fg*alpha[...,None]
    weight=alpha.sum();centroid=np.array([(alpha*xx).sum(),(alpha*yy).sum()])/weight
    transform=np.float32([[1,0,half-centroid[0]],[0,1,half-centroid[1]]])
    alpha=cv2.warpAffine(alpha,transform,(size,size))
    premul=cv2.warpAffine(premul.astype(np.float32),transform,(size,size))
    return premul,alpha,patch.astype('uint8'),bg.astype('uint8')

def transform_sprite(premul,alpha,angle):
    n=len(alpha);m=cv2.getRotationMatrix2D(((n-1)/2,(n-1)/2),-angle,1)
    return cv2.warpAffine(premul,m,(n,n)),cv2.warpAffine(alpha,m,(n,n))

def composite(plate,xy,premul,alpha,delta,exposure):
    n=len(alpha);half=(n-1)/2
    shifted=np.zeros_like(premul);coverage=np.zeros_like(alpha)
    # Integrate a continuous footprint, never draw separate ghost-ball copies.
    for offset in np.linspace(-.5,.5,17):
        m=np.float32([[1,0,float(delta[0]*exposure*offset)],[0,1,float(delta[1]*exposure*offset)]])
        shifted+=cv2.warpAffine(premul,m,(n,n))/17
        coverage+=cv2.warpAffine(alpha,m,(n,n))/17
    left,top=np.floor(np.asarray(xy)-half).astype(int)
    m=np.float32([[1,0,float(xy[0]-half-left)],[0,1,float(xy[1]-half-top)]])
    shifted=cv2.warpAffine(shifted,m,(n,n));coverage=cv2.warpAffine(coverage,m,(n,n))
    result=plate.copy();roi=plate[top:top+n,left:left+n].astype(float)
    if roi.shape!=shifted.shape:raise ValueError('Sprite outside image')
    result[top:top+n,left:left+n]=np.rint(np.clip(roi*(1-coverage[...,None])+shifted,0,255)).astype('uint8')
    support=np.zeros(plate.shape[:2],'uint8');support[top:top+n,left:left+n]=(coverage>1e-5).astype('uint8')
    if np.any(np.any(result!=plate,axis=2)&(support==0)):raise ValueError('Background altered outside alpha footprint')
    return result,support

def main():
    OUT.mkdir(exist_ok=True)
    comparison=json.loads((DATA/'ff_to_sl_model_preview/comparison.json').read_text())
    start,end=comparison['source_window_s']
    rows=sorted([r for r in json.loads((DATA/'dataset.json').read_text())['rows'] if r['sample_id']=='FF_02'],key=lambda r:r['decoded_pts'])
    times=np.array([r['decoded_pts'] for r in rows]);hints=np.array([[r['x'],r['y']] for r in rows])
    cap=cv2.VideoCapture(str(ROOT/rows[0]['source_video']));source=[];pts=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if start-1e-5<=t<=end+1e-5:source.append(frame);pts.append(t)
        if t>end+.02:break
    cap.release();pts=np.array(pts);fps=1/np.median(np.diff(pts));slowfps=fps/5
    n=len(source);u=(pts-start)/(end-start);grid=np.linspace(0,1,81)
    tracks={}
    for case in CASES:
        t=np.load(DATA/'ff_to_sl_model_preview'/case/'tracks.npy')[:,6]
        native=np.column_stack([t[:,0]/.65,(t[:,1]-6)/.65])
        tracks[case]=np.column_stack([np.interp(u,grid,native[:,k]) for k in range(2)])
    centers=[]
    for frame,t in zip(source,pts):
        hint=np.array([np.interp(t,times,hints[:,k]) for k in range(2)])
        if t<=3.4361: center=component_center(frame,hint)
        else:
            # Ball darkens beside the helmet; a brightness detector would select
            # the helmet highlight. End anchors were reviewed in native crops.
            anchors=np.array([[3.4360222,654.34,280.56],[3.4860778,660.3,280.4],[3.5194444,663.4,283.4]])
            center=np.array([np.interp(t,anchors[:,0],anchors[:,k]) for k in [1,2]])
        centers.append(center)
    centers=np.array(centers)
    bank=[];banksrc=[]
    # Visual matte audit rejects later sprites containing helmet highlights.
    # Only the first five unobstructed native frames form this baseline bank.
    for i in range(min(5,n)):
        p,a,raw,bg=sprite(source[i],centers[i]);bank.append((p,a));banksrc.append((raw,bg))
        d=OUT/'sprites';d.mkdir(exist_ok=True)
        fg=np.divide(p,np.maximum(a[...,None],.001));rgba=np.dstack([np.clip(fg,0,255).astype('uint8'),np.rint(a*255).astype('uint8')])
        cv2.imwrite(str(d/f'{i:03d}.png'),rgba)
    plates=[];records=[]
    for i,(frame,center) in enumerate(zip(source,centers)):
        mask=np.zeros(frame.shape[:2],'uint8');cv2.circle(mask,tuple(np.rint(center).astype(int)),10,255,-1)
        plate=cv2.inpaint(frame,mask,3,cv2.INPAINT_TELEA);plates.append(plate)
        for name,image in [('source_frames',frame),('clean_plates',plate),('removal_masks',mask)]:
            d=OUT/name;d.mkdir(exist_ok=True);cv2.imwrite(str(d/f'{i:03d}.png'),image)
        records.append({'frame':i,'source_pts_s':float(pts[i]),'old_ball_native_xy':center.tolist(),'cleanup':'bright-component hint' if pts[i]<=3.4361 else 'visually reviewed end anchors'})
    sourcevel=np.gradient(centers,axis=0)
    for variant,exposure in [('observed_appearance',0.),('exposure_blur',1.5)]:
        for case in CASES:
            folder=OUT/variant/case/'frames';folder.mkdir(parents=True,exist_ok=True)
            velocity=np.gradient(tracks[case],axis=0)
            for i,(plate,xy) in enumerate(zip(plates,tracks[case])):
                # Smoothly traverse uncontaminated real-video sprites. No new
                # appearance is synthesized or claimed as learned.
                pos=u[i]*(len(bank)-1);lo=int(np.floor(pos));hi=min(lo+1,len(bank)-1);w=pos-lo
                p=(1-w)*bank[lo][0]+w*bank[hi][0];a=(1-w)*bank[lo][1]+w*bank[hi][1]
                srcv=(1-w)*sourcevel[lo]+w*sourcevel[hi]
                v=velocity[i]
                angle=np.degrees(np.arctan2(v[1],v[0])-np.arctan2(srcv[1],srcv[0]))
                # Both outputs share the identical first-frame appearance.
                if i==0:
                    v=np.gradient(tracks[CASES[0]],axis=0)[0]
                    angle=np.degrees(np.arctan2(v[1],v[0])-np.arctan2(srcv[1],srcv[0]))
                p,a=transform_sprite(p,a,angle)
                speed=np.linalg.norm(v);delta=v*min(1,5/max(speed,1e-6))
                result,support=composite(plate,xy,p,a,delta,exposure)
                cv2.imwrite(str(folder/f'{i:03d}.png'),result)
                sd=OUT/variant/case/'alpha_support';sd.mkdir(exist_ok=True);cv2.imwrite(str(sd/f'{i:03d}.png'),support*255)
            encode(folder,folder.parent/'normal_speed.mp4',fps)
            encode(folder,folder.parent/'slow_motion.mp4',slowfps)
        first=[cv2.imread(str(OUT/variant/c/'frames/000.png')) for c in CASES]
        if not np.array_equal(*first):raise ValueError('First frames differ')
        for speed in ['normal_speed','slow_motion']:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(OUT/variant/CASES[0]/f'{speed}.mp4'),'-i',str(OUT/variant/CASES[1]/f'{speed}.mp4'),'-filter_complex','hstack=inputs=2','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',str(OUT/variant/f'{speed}_comparison.mp4')],check=True)
    # Source / white-dot baseline / real appearance / exposure-integrated patch.
    tiles=[]
    for i in [0,5,10,15,20,n-1]:
        oldi=int(round(u[i]*80))
        old=cv2.imread(str(DATA/'compositor_test/source_motion/original_ff/frames'/f'{oldi:03d}.png'))
        old=cv2.resize(old[6:474],(1280,720))
        panels=[source[i],old,cv2.imread(str(OUT/'observed_appearance/original_ff/frames'/f'{i:03d}.png')),cv2.imread(str(OUT/'exposure_blur/original_ff/frames'/f'{i:03d}.png'))]
        row=[]
        for label,panel in zip(['Real source','Old white-dot baseline','Real matte transferred','Exposure blur (assumed)'],panels):
            crop=cv2.resize(panel[250:355,580:700],(360,315),interpolation=cv2.INTER_NEAREST)
            crop=cv2.copyMakeBorder(crop,26,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(crop,f'{label} / {i}',(4,18),0,.45,(255,255,255),1,cv2.LINE_AA);row.append(crop)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(OUT/'appearance_audit.jpg'),np.vstack(tiles))
    # Enlarged pair for assessment, without annotations inside generated frames.
    d=OUT/'exposure_blur/zoom_frames';d.mkdir(exist_ok=True)
    for i in range(n):
        pair=[cv2.resize(cv2.imread(str(OUT/'exposure_blur'/c/'frames'/f'{i:03d}.png'))[240:380,560:740],(540,420)) for c in CASES]
        cv2.imwrite(str(d/f'{i:03d}.png'),np.hstack(pair))
    encode(d,OUT/'exposure_blur/zoom_slow_motion.mp4',slowfps)
    verification={}
    for path in OUT.glob('*/*/*.mp4'):
        cap=cv2.VideoCapture(str(path));count=0
        while cap.read()[0]:count+=1
        cap.release()
        if count!=n:raise ValueError('Video frame count mismatch')
        verification[str(path.relative_to(OUT))]={'decoded_frames':count}
    report={'method':'real-video RGBA patch transfer, not learned generation','source':'FF_02','frames':n,'source_fps_estimate':float(fps),'source_duration_s':float(end-start),'slowdown':5,'sprite_bank_frames':len(bank),'shared_background_exact_outside_ball_support':True,'same_first_frame':True,'exposure_variant_assumed_frame_intervals':1.5,'records':records,'videos':verification,'limitations':['Matte estimated from inpainted background, not ground truth.','Empirical source blur preserved; additional exposure is a visual assumption.','5 unobstructed sprites are resampled; no pitch-type appearance learning.','End source ball cleanup uses reviewed anchors; helmet reconstruction may smear.','No depth ordering or catch/contact modeled; retained visible flight only.','FF route is TrackNet-derived; SL route is a geometric candidate.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','videos']},indent=2))

if __name__=='__main__':main()
