"""Frozen Neural ODE initial-condition edits and RGB condition construction.

All appearances are real-sprite compositing, never neural video generation.
"""
import json
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'baseb-stage4-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from export_stage1 import dump, encode_images, probe, sha256
from ode_model import BallODE, integrate
from prepare_stage3 import project

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
EDIT, DELIVERY = EXP/'edits', EXP/'delivery/stage4'


def rollout(model, state, kind, times, step=1/240):
    code = torch.tensor([[1.,0.] if kind=='FF' else [0.,1.]],dtype=torch.float64)
    with torch.no_grad():
        result=integrate(model,torch.as_tensor(np.asarray(state)[None],dtype=torch.float64),code,
                         torch.as_tensor(np.asarray(times)[None],dtype=torch.float64),step)[0]
        acceleration=model(result,code.expand(len(result),-1))[:,3:]
    return result.numpy(),acceleration.numpy()


def composite(plate, xy, sprite, scale):
    # Preserve the empirical sprite blur; no unmeasured exposure is added.
    rgba=cv2.resize(sprite,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    alpha=rgba[:,:,3].astype(float)/255
    yy,xx=np.indices(alpha.shape);center=np.array([(xx*alpha).sum(),(yy*alpha).sum()])/alpha.sum()
    h,w=plate.shape[:2]
    M=np.float32([[1,0,xy[0]-center[0]],[0,1,xy[1]-center[1]]])
    a=cv2.warpAffine(alpha.astype(np.float32),M,(w,h))
    premultiplied=cv2.warpAffine(rgba[:,:,:3].astype(np.float32)*alpha[:,:,None],M,(w,h))
    result=np.rint(np.clip(plate*(1-a[:,:,None])+premultiplied,0,255)).astype(np.uint8)
    support=(a>1e-5).astype(np.uint8)*255
    assert not np.any(np.any(result!=plate,axis=2)&(support==0))
    return result,support


def check_video(path,times,duration,size):
    meta=probe(path);tb=Fraction(meta['streams'][0]['time_base'])
    pts=np.array([float(int(f.get('pts',f['best_effort_timestamp']))*tb) for f in meta['frames']])
    assert len(pts)==len(times) and (meta['streams'][0]['width'],meta['streams'][0]['height'])==tuple(size)
    error=float(max(abs(pts-times)));assert error<1/90000*1.1
    assert abs(float(meta['streams'][0]['duration'])-duration)<1/90000*1.1
    return {'path':str(path),'frames':len(pts),'size':size,'pts_max_error_s':error,'duration_s':float(meta['streams'][0]['duration'])}


def main():
    torch.set_num_threads(1)
    for p in [EDIT,DELIVERY]:p.mkdir(parents=True,exist_ok=True)
    data=json.loads((EXP/'training/prepared_data.json').read_text())
    sample=next(s for s in data['samples'] if s['sample_id']=='FF_02')
    sl=next(s for s in data['samples'] if s['sample_id']=='SL_02')
    checkpoint_path=EXP/'training/model.pt';weights_hash=sha256(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=BallODE(checkpoint['normalization']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=sample['rows'][sample['prediction_start_row']:sample['prediction_start_row']+16]
    assert len(rows)==16 and all(r['use_for_observation_loss'] for r in rows)
    initial=np.array(rows[0]['position_m']+rows[0]['velocity_mps'])
    sl_velocity=sl['rows'][sl['prediction_start_row']]['velocity_mps']
    defaults={'source_sample':'FF_02','common_position_m':initial[:3].tolist(),
        'camera_source':'training/prepared_data.json FF_02 frozen camera',
        'frame_count':16,'primary_cases':['ff_base','sl_variant'],
        'cases':[{'id':'ff_base','pitch_type':'FF','velocity_mps':initial[3:].tolist(),'edit':'original FF_02 prefix-end state'},
                 {'id':'sl_variant','pitch_type':'SL','velocity_mps':sl_velocity,'edit':'SL_02 prefix-end velocity + SL condition; same position'},
                 {'id':'ff_slower','pitch_type':'FF','velocity_mps':(initial[3:]*.98).tolist(),'edit':'speed magnitude -2%; direction unchanged'},
                 {'id':'ff_rightward','pitch_type':'FF','velocity_mps':(initial[3:]+[.35,0,0]).tolist(),'edit':'vx +0.35 m/s'},
                 {'id':'ff_upward','pitch_type':'FF','velocity_mps':(initial[3:]+[0,0,.30]).tolist(),'edit':'vz +0.30 m/s'}]}
    config_path=EDIT/'edit_config.json'
    if not config_path.exists():dump(config_path,defaults)
    config=json.loads(config_path.read_text());assert config['source_sample']=='FF_02' and config['frame_count']==16
    camera=sample['camera'];camera_hash=sha256(EXP/'training/camera_transfer.json')
    times=np.array([r['video_pts_s']-rows[0]['video_pts_s'] for r in rows])
    last_duration=sample['rows'][sample['prediction_start_row']+16]['video_pts_s']-rows[-1]['video_pts_s']
    duration=float(times[-1]+last_duration)
    all_train=np.array([r['position_m']+r['velocity_mps'] for s in data['samples'] if s['split']=='train' for r in s['rows']])
    domains={kind:np.array([r['position_m']+r['velocity_mps'] for s in data['samples'] if s['split']=='train' and s['pitch_type']==kind for r in s['rows']]) for kind in ['FF','SL']}
    seeds={kind:np.array([s['rows'][s['prediction_start_row']]['velocity_mps'] for s in data['samples'] if s['split']=='train' and s['pitch_type']==kind]) for kind in ['FF','SL']}
    case_data={};records=[]
    for case in config['cases']:
        state=np.array(config['common_position_m']+case['velocity_mps']);kind=case['pitch_type']
        assert np.all(state>=domains[kind].min(0)) and np.all(state<=domains[kind].max(0)), 'Initial state outside pitch-specific training component envelope'
        assert np.all(state[3:]>=seeds[kind].min(0)) and np.all(state[3:]<=seeds[kind].max(0)), 'Velocity outside prefix seed envelope'
        states,accel=rollout(model,state,kind,times);refined,_=rollout(model,state,kind,times,1/480)
        assert np.isfinite(states).all() and np.array_equal(states[0],state)
        assert max(np.linalg.norm(states[:,:3]-refined[:,:3],axis=1))<1e-4
        uv=project(camera,states[:,:3]);depth=(states[:,:3]@np.array(camera['R_world_to_camera']).T+camera['T_world_to_camera_m'])[:,2]
        diameter=np.array(camera['K'])[0,0]*.073/depth
        screen_velocity=np.gradient(uv,times,axis=0,edge_order=2)
        out={'case_id':case['id'],'pitch_type':kind,'edit_description':case['edit'],
             'original_initial_state':initial.tolist(),'edited_initial_state':state.tolist(),
             'delta_state':(state-initial).tolist(),'edited_fields':[k for k,d in zip(['x','y','z','vx','vy','vz'],state-initial) if d!=0]+(['pitch_type'] if kind!='FF' else []),
             'initial_speed_mps':float(np.linalg.norm(state[3:])),'source_velocity_sample':'SL_02' if case['id']=='sl_variant' else 'FF_02',
             'camera':camera,'same_position_and_camera':True,'times_s':times.tolist(),
             'training_domain_check':{'initial_state_in_pitch_component_envelope':True,'velocity_in_pitch_prefix_seed_envelope':True,
                'rollout_outside_pitch_component_envelope_frames':int(np.any((states<domains[kind].min(0))|(states>domains[kind].max(0)),axis=1).sum()),
                'interpretation':'component bounds only; joint state/condition combinations are edited counterfactuals, not independently validated physical trajectories'},
             'rows':[{'frame_index':r['frame_index'],'source_video_pts_s':r['video_pts_s'],'elapsed_s':float(times[i]),
                      'position_m':states[i,:3].tolist(),'velocity_mps':states[i,3:].tolist(),'net_acceleration_mps2':accel[i].tolist(),
                      'state_source':'edited_initial' if i==0 else 'predicted','event_type':'none','uv_px':uv[i].tolist(),
                      'nominal_diameter_px':float(diameter[i]),'screen_velocity_px_s':screen_velocity[i].tolist(),
                      'screen_direction_deg':float(np.degrees(np.arctan2(screen_velocity[i,1],screen_velocity[i,0]))),
                      'support_bbox_original_px':[int(np.floor(uv[i,0]-14)),int(np.floor(uv[i,1]-14)),int(np.ceil(uv[i,0]+14)),int(np.ceil(uv[i,1]+14))]}
                     for i,r in enumerate(rows)]}
        case_data[case['id']]={'states':states,'uv':uv,'diameter':diameter,'record':out}
        dump(EDIT/f"{case['id']}_trajectory.json",out);records.append(out)
    base=case_data['ff_base']
    comparisons={}
    for name,c in case_data.items():
        if name=='ff_base':continue
        separation=np.linalg.norm(c['uv']-base['uv'],axis=1)
        comparisons[name]={'comparison':'ff_base -> '+name,'frame_count':len(times),'endpoint_separation_px':float(separation[-1]),
            'max_separation_px':float(separation.max()),'endpoint_separation_m':float(np.linalg.norm(c['states'][-1,:3]-base['states'][-1,:3])),
            'endpoint_delta_position_m':(c['states'][-1,:3]-base['states'][-1,:3]).tolist(),
            'only_initial_speed_changed':name=='ff_slower','same_pitch_type':c['record']['pitch_type']=='FF',
            'separation_px_by_frame':separation.tolist(),'not_a_real_world_accuracy_metric':True}
    # One shared removal pass. Telea is a provisional condition background.
    shared=DELIVERY/'shared_background';shared.mkdir(exist_ok=True)
    masks=DELIVERY/'old_ball_masks';masks.mkdir(exist_ok=True)
    backgrounds=[]
    for i,r in enumerate(rows):
        source=cv2.imread(r['source_image']);mask=np.zeros(source.shape[:2],np.uint8)
        cv2.circle(mask,(round(r['u_px']),round(r['v_px'])),12,255,-1)
        bg=cv2.inpaint(source,mask,3,cv2.INPAINT_TELEA)
        assert not np.any(np.any(source!=bg,axis=2)&(mask==0))
        cv2.imwrite(str(shared/f'{i:04d}.png'),bg);cv2.imwrite(str(masks/f'{i:04d}.png'),mask);backgrounds.append(bg)
    all_uv=np.vstack([c['uv'] for c in case_data.values()]+[np.array([[r['u_px'],r['v_px']] for r in rows])])
    # Square ROI avoids aspect distortion for 384x384 RGB conditioning.
    mid=(all_uv.min(0)+all_uv.max(0))/2;side=max(144,int(np.ceil(np.max(all_uv.max(0)-all_uv.min(0))+56)))
    side=int(np.ceil(side/8)*8);left,top=np.floor(mid-side/2).astype(int)
    assert 0<=left and 0<=top and left+side<=1280 and top+side<=720
    roi={'left':int(left),'top':int(top),'width':side,'height':side,'output_size':[384,384],
         'original_to_condition':[[384/side,0,-left*384/side],[0,384/side,-top*384/side],[0,0,1]],
         'source':'union of all predicted trajectories and source-ball removal region + margin'}
    sprite_path=ROOT/'data/tracknet_ff_sl/appearance_transfer/sprites/000.png'
    sprite=cv2.imread(str(sprite_path),cv2.IMREAD_UNCHANGED);assert sprite.shape[2]==4
    a=sprite[:,:,3]/255.;yy,xx=np.indices(a.shape);active=a>.25
    # Nominal projected size relative to measured high-alpha sprite footprint.
    sprite_diameter=float(max(np.ptp(xx[active])+1,np.ptp(yy[active])+1))
    video_checks=[]
    for name,c in case_data.items():
        folder=DELIVERY/name;full=folder/'composite_frames';condition=folder/'rgb_conditions';support=folder/'alpha_support'
        for d in [full,condition,support]:d.mkdir(parents=True,exist_ok=True)
        rendered=[]
        for i,(bg,uv,diameter) in enumerate(zip(backgrounds,c['uv'],c['diameter'])):
            scale=diameter/sprite_diameter
            if i==0:scale=base['diameter'][0]/sprite_diameter
            frame,alpha=composite(bg,uv,sprite,scale)
            assert not np.any((alpha>0)&((np.indices(alpha.shape)[1]<left)|(np.indices(alpha.shape)[1]>=left+side)|(np.indices(alpha.shape)[0]<top)|(np.indices(alpha.shape)[0]>=top+side)))
            cv2.imwrite(str(full/f'{i:04d}.png'),frame)
            cv2.imwrite(str(condition/f'{i:03d}.png'),cv2.resize(frame[top:top+side,left:left+side],(384,384),interpolation=cv2.INTER_CUBIC))
            cv2.imwrite(str(support/f'{i:04d}.png'),alpha);rendered.append(frame)
            record=c['record']['rows'][i];record['condition_uv_px']=((uv-[left,top])*384/side).tolist()
            sy,sx=np.nonzero(alpha);record['support_bbox_original_px']=[int(sx.min()),int(sy.min()),int(sx.max()+1),int(sy.max()+1)]
            record['condition_screen_velocity_px_s']=(np.asarray(record['screen_velocity_px_s'])*384/side).tolist()
            record['rgb_condition_path']=str(condition/f'{i:03d}.png');record['source_sprite_path']=str(sprite_path)
            record['sprite_scale']=float(scale);record['alpha_support_path']=str(support/f'{i:04d}.png')
        for factor,filename in [(1,'normal_speed.mp4'),(5,'slow_5x.mp4')]:
            encode_images(full,times*factor,last_duration*factor,folder/filename)
            video_checks.append(check_video(folder/filename,times*factor,duration*factor,[1280,720]))
        c['rendered']=rendered
        dump(EDIT/f'{name}_trajectory.json',c['record'])
    assert all(np.array_equal(base['rendered'][0],c['rendered'][0]) for c in case_data.values())
    # Labels belong only on comparison previews, never on RGB condition frames.
    for target in ['sl_variant','ff_slower','ff_rightward','ff_upward']:
        folder=DELIVERY/(target+'_comparison_frames');folder.mkdir(exist_ok=True)
        for i in range(16):
            panels=[]
            for name in ['ff_base',target]:
                frame=case_data[name]['rendered'][i]
                crop=cv2.resize(frame[top:top+side,left:left+side],(480,480),interpolation=cv2.INTER_CUBIC)
                crop=cv2.copyMakeBorder(crop,52,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                cv2.putText(crop,name,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(crop,f't={times[i]:.4f}s / COMPOSITE CONDITION PREVIEW',(12,43),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1,cv2.LINE_AA)
                panels.append(crop)
            cv2.imwrite(str(folder/f'{i:04d}.png'),np.hstack(panels))
        out=DELIVERY/f'{target}_comparison_slow_5x.mp4'
        encode_images(folder,times*5,last_duration*5,out);video_checks.append(check_video(out,times*5,duration*5,[960,532]))
    case_colors={
        'ff_base':'#1f77b4',
        'sl_variant':'#ff7f0e',
        'ff_slower':'#2ca02c',
        'ff_rightward':'#d62728',
        'ff_upward':'#9467bd',
    }
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for name,c in case_data.items():
        uv=c['uv'];states=c['states'];color=case_colors[name]
        axes[0].plot(uv[:,0],uv[:,1],label=name,color=color)
        axes[1].plot(states[:,1],states[:,2],label=name,color=color)
        if name!='ff_base':
            axes[2].plot(times,comparisons[name]['separation_px_by_frame'],label=name,color=color)
    axes[0].invert_yaxis();axes[0].set(xlabel='image u (px)',ylabel='image v (px)',title='Same effective camera')
    axes[1].invert_xaxis();axes[1].set_box_aspect(.34);axes[1].set_anchor('C')
    axes[1].set(xlabel='world y (m)',ylabel='world z (m)',title='Learned 3D predictions (vertical scale expanded)')
    axes[2].set(xlabel='elapsed from common initial frame (s)',ylabel='distance from FF base (px)',title='Projected edit separation')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=5,fontsize=8,frameon=False,bbox_to_anchor=(.5,.995))
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(EDIT/'edit_comparison.png',dpi=160);plt.close(fig)
    sheet=Image.new('RGB',(5*250,4*280),'white');draw=ImageDraw.Draw(sheet)
    for col,(name,c) in enumerate(case_data.items()):
        for row,i in enumerate([0,5,10,15]):
            crop=Image.fromarray(cv2.cvtColor(c['rendered'][i],cv2.COLOR_BGR2RGB)).crop((left,top,left+side,top+side)).resize((240,240))
            sheet.paste(crop,(col*250,row*280+35));draw.text((col*250+5,row*280+8),f'{name} / t={times[i]:.3f}s',fill='black')
    sheet.save(DELIVERY/'conditions_contact_sheet.png')
    manifest={'stage':4,'status':'completed_edited_trajectories_and_rgb_conditions','primary_cases':config['primary_cases'],
        'controlled_speed_cases':['ff_slower','ff_rightward','ff_upward'],'common_initial_position_m':config['common_position_m'],
        'camera':camera,'camera_fixed':True,'model_sha256':weights_hash,'roi':roi,'source_sample':'FF_02',
        'source_frame_ids':[r['frame_index'] for r in rows],'source_pts_s':[r['video_pts_s'] for r in rows],
        'elapsed_times_s':times.tolist(),'duration_s':duration,'last_frame_duration_s':float(last_duration),
        'source_frame_selection':'first 16 actual frames starting at stage3 prefix-end state, not release',
        'world_units':'m, m/s, m/s²','nominal_ball_diameter_m':.073,'comparisons':comparisons,
        'shared_background':{'method':'one OpenCV Telea removal pass, radius 12px manual center mask',
            'path':str(shared),'role':'provisional RGB conditioning background; moving cleanup artifacts may remain',
            'background_hashes':[sha256(shared/f'{i:04d}.png') for i in range(16)]},
        'appearance':{'method':'reuse one real RGBA sprite, size by nominal projected diameter, fractional-position alpha composite',
            'sprite_path':str(sprite_path),'sprite_sha256':sha256(sprite_path),'empirical_blur_preserved':True,
            'added_motion_blur':False,'camera_exposure_measured':False,'pitch_specific_appearance_learned':False},
        'consumer':'SparseCtrl RGB image sequences; JSON centers/masks are preprocessing metadata, not direct independent model modalities',
        'not_neural_video_generation':True,'new_real_world_ground_truth_available':False,
        'limitations':['Counterfactual edited state combinations only; bounds do not prove physical correctness.',
                       'Effective camera ambiguity remains; use current local scene only.',
                       'Initial time is prefix end, not release; no catch/contact or automatic occlusion handling.',
                       'Telea background provisional; stage5 must inspect and refine if artifacts affect generation.'],
        'cases':[c['record'] for c in case_data.values()]}
    assert sha256(checkpoint_path)==weights_hash and sha256(EXP/'training/camera_transfer.json')==camera_hash
    for path,expected in data['source_hashes'].items():assert sha256(Path(path))==expected,path
    dump(EDIT/'stage4_manifest.json',manifest)
    dump(EXP/'reports/stage4_validation.json',{'status':'numerical_and_interface_checks_passed',
        'source_and_training_artifacts_unchanged':True,'same_initial_position_camera_and_times':True,
        'all_initial_states_and_velocities_within_pitch_component_bounds':True,'same_composite_first_frame':True,
        'no_forced_endpoints':True,'no_posthoc_trajectory_adjustment':True,'outside_alpha_support_background_unchanged':True,
        'all_supports_within_shared_roi':True,'rgb_condition_size':[384,384],'frames_per_case':16,
        'case_count':len(case_data),'video_checks':video_checks,'visual_review':'pending'})
    lines=['# 阶段四：初始条件编辑与生成条件','',
        '状态：固定阶段三权重与 FF_02 有效相机，已完成 5 条编辑球路及 RGB 条件；没有运行视频生成网络。','',
        '共享 FF_02 前段结束位置和 16 个实际视频时间（解码帧 191–206）。这不是出手位置/时刻。原型 FF 的初速度来自 FF_02 参考状态；SL 主球路使用 SL_02 前段速度但共享 FF 位置，因此两者同时改变球种和速度，不将差异全归因于球种。','',
        '| 球路 | 条件编辑 | 初始速度大小 m/s | 相对 FF 末帧距离 px |',
        '|---|---|---:|---:|']
    for name,c in case_data.items():lines.append(f"| {name} | {c['record']['edit_description']} | {c['record']['initial_speed_mps']:.3f} | {comparisons[name]['endpoint_separation_px']:.2f} |" if name!='ff_base' else f"| {name} | 原型 FF | {c['record']['initial_speed_mps']:.3f} | 0 |")
    lines += ['', '所有初始状态均落在对应球种训练数据的逐分量范围内，速度也落在对应前段种子速度范围内；这不是联合分布覆盖或真实物理验证。改动后的球路没有真实反事实真值，球路间距离是编辑差异，不能称为预测精度。模型自行积分产生不同终点，没有强制终点匹配。','',
        f"共同窗口首末间隔 {times[-1]:.6f} 秒，含末帧总时长 {duration:.6f} 秒。所有输出共享相机与采样时刻，正常速度视频保持原 PTS，慢放标明 5 倍。",'',
        f"共同方形 ROI：left={left}, top={top}, side={side}，缩放到 384×384，变换保存在 manifest。每条保存 16 张无字幕 RGB 条件、原尺寸合成帧、alpha 支持区域及逐帧位置/速度、投影、尺度和屏幕速度。",'',
        '原球只处理一次：按当前核对过的人工球心生成半径 12 px 掩膜，Telea 局部修复，全部候选共享同一背景。它是本地条件准备用的临时背景，可能有移动修复痕迹；阶段五需要根据画面检查决定是否进一步修复。',
        '复用已审查的真实 RGBA 球影素材，按名义 0.073 m 球径的相机投影尺度放置。保留素材原有模糊，不添加未测量的曝光。球影、模糊和遮挡没有被运动网络学习；当前未做接球/人物遮挡建模。',
        '**预览视频属于球影合成与生成条件展示，不是 AnimateDiff/SparseCtrl 输出。** SparseCtrl 接收 RGB 图像序列，球心 JSON 和 alpha 掩膜只服务于条件构造。所有候选首帧的合成像素完全相同。','',
        '- [编辑配置](../edits/edit_config.json) · [完整接口与编辑记录](../edits/stage4_manifest.json)',
        '- [三维/投影差异图](../edits/edit_comparison.png)',
        '- [条件检查图](../delivery/stage4/conditions_contact_sheet.png)',
        '- [FF/SL 主对照，5 倍慢放](../delivery/stage4/sl_variant_comparison_slow_5x.mp4)',
        '- [同球种速度大小对照，5 倍慢放](../delivery/stage4/ff_slower_comparison_slow_5x.mp4)',
        '- [查看页](../delivery/stage4/index.html) · [验证](stage4_validation.json) · [视觉检查](stage4_visual_review.md)','',
        '复现：修改 edit_config.json 中的初速度或球种后运行（脚本检查本轮范围）：','',
        '```sh','bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/edit_stage4.py','```','',
        '阶段五待执行：复用主 FF/SL 的 RGB 条件与共用背景运行冻结的视频生成器，保存原始生成结果并检查球路遵循、丢球、旧球残留、修复痕迹与遮挡。','']
    (EXP/'reports/stage4_report.md').write_text('\n'.join(lines))
    cards=''.join(f'<section><h2>FF base vs {n}</h2><video controls loop src="{n}_comparison_slow_5x.mp4"></video></section>' for n in comparisons)
    (DELIVERY/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阶段四：初速度编辑</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#19212b}section{background:white;padding:20px;margin:20px 0;border-radius:12px}video,img{max-width:100%}</style><h1>固定相机下的初始条件编辑</h1><p>所有对照均为 5 倍慢放。左：原型 FF；右：编辑球路。相同初始位置、视频时间和背景。</p><p>这是 Neural ODE 球路配合真实球影合成的 RGB 条件预览，尚未运行视频生成网络。FF/SL 主对照同时改变球种和初速度；其余对照只改变 FF 初速度。</p>'+cards+'<p><a href="conditions_contact_sheet.png">条件帧检查图</a></p></html>')
    print(json.dumps({'status':'stage4_complete_pending_visual_review',
        'endpoint_separation_px':{k:v['endpoint_separation_px'] for k,v in comparisons.items()},'roi':roi},indent=2),flush=True)


if __name__=='__main__':main()
