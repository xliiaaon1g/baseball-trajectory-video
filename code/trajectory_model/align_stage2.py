"""Build Statcast reference states and calibrate effective broadcast cameras.

Read-only source data; full-span calibration and a separate prefix-only check.
The fitted trajectory is reference geometry, never a Neural ODE prediction.
"""
import csv
import json
import math
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'baseb-stage2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import stage2_geometry as geom
from export_stage1 import dump, encode_images, probe, sha256

ROOT, EXP = geom.ROOT, geom.EXP
ALIGN = EXP / 'alignment'
DELIVERY = EXP / 'delivery/stage2'
REPORTS = EXP / 'reports'
SOURCES = {
    'statcast_fields': 'https://baseballsavant.mlb.com/csv-docs',
    'coordinate_reference': 'https://physics.csuchico.edu/baseball/resources/POBActivities/pitchfx/ws2012.shtml',
    'field_geometry': 'https://mktg.mlbstatic.com/mlb/official-information/2025-official-baseball-rules.pdf',
    'pnp': 'https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html',
}


def stats(errors):
    errors = np.asarray(errors)
    return {'count': len(errors), 'median_px': float(np.median(errors)),
            'p90_px': float(np.quantile(errors, .9)), 'max_px': float(np.max(errors)),
            'rmse_px': float(np.sqrt(np.mean(errors**2)))}


def visible_ball_core(image, row):
    if not row['use_for_observation_loss']:
        return None
    x, y = round(row['u_px']), round(row['v_px'])
    patch=image[y-14:y+15,x-14:x+15]
    if patch.shape[:2]!=(29,29):
        return None
    bright=((patch.min(2)>=180)&((patch.max(2).astype(int)-patch.min(2).astype(int))<=55)).astype(np.uint8)
    count, labels, components, centers=cv2.connectedComponentsWithStats(bright)
    choices=[j for j in range(1,count) if np.linalg.norm(centers[j]-[14,14])<=5
             and components[j,cv2.CC_STAT_AREA]>=12]
    if not choices:
        return None
    j=min(choices,key=lambda j:np.linalg.norm(centers[j]-[14,14]))
    xx,yy,w,h,area=components[j]
    if xx<=0 or yy<=0 or xx+w>=29 or yy+h>=29 or area>=250 or max(w,h)/min(w,h)>2.5:
        return None
    return {'equivalent_diameter_px':float(math.sqrt(4*area/math.pi)),
            'bbox_wh_px':[int(w),int(h)],'bright_area_px':int(area),
            'method':'near-center bright component, min RGB 180, channel range <=55; rejects borders/tiny fragments',
            'interpretation':'visible bright-core size estimate; not full blurred silhouette or independent size truth'}


def project_rows(q, sid, rows):
    t = np.array([r['video_pts_s'] for r in rows]) - (q * geom.scale)[7+geom.SIDS.index(sid)]
    p, v, a, _ = geom.refs[sid]
    X = p + t[:, None] * v + .5 * t[:, None]**2 * a
    return t, X, v + t[:, None]*a, geom.project(q, X, sid)


def camera_record(q, sid, fit_scope='all'):
    K, R, T = geom.camera(q)
    if sid == 'SL_02':
        A = geom.H[:2, :2]
        s = np.sqrt(np.linalg.det(A))
        image_rotation = np.eye(3)
        image_rotation[:2, :2] = A/s
        principal = A @ K[:2, 2] + geom.H[:2, 2]
        K = np.array([[s*K[0, 0], 0, principal[0]], [0, s*K[1, 1], principal[1]], [0, 0, 1]])
        R, T = image_rotation @ R, image_rotation @ T
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8) and abs(np.linalg.det(R)-1) < 1e-8
    rows = geom.rows_for(sid, 'all')
    _, X, _, uv = project_rows(q, sid, rows)
    cam = X @ R.T + T
    projected = (cam @ K.T)[:, :2] / cam[:, 2:]
    assert np.all(cam[:, 2] > 0) and np.max(abs(projected-uv)) < 1e-8
    return {'sample_id': sid, 'camera_segment_id': f'{sid}_fixed_flight', 'K': K.tolist(),
            'R_world_to_camera': R.tolist(), 'T_world_to_camera_m': T.tolist(),
            'camera_center_world_m': (-R.T @ T).tolist(), 'image_size': [1280, 720],
            'distortion': [0, 0, 0, 0, 0], 'distortion_source': 'fixed_zero_assumption',
            'crop_resize_transform': np.eye(3).tolist(),
            'calibration_source': 'effective_camera_estimated_from_static_field_and_train_pitch_reference',
            'calibration_samples': geom.SIDS, 'calibration_frames': {
                s: [r['frame_index'] for r in geom.rows_for(s, fit_scope)] for s in geom.SIDS},
            'official_hardware_calibration': False,
            'image_similarity_from_FF_02': (np.eye(3) if sid=='FF_02' else geom.H).tolist()}


def verify_background():
    mask = np.zeros((720, 1280), np.uint8)
    mask[160:285, 20:1250] = 255
    mask[180:285, 500:820] = 0
    orb = cv2.ORB_create(nfeatures=2500)
    first = {s: cv2.imread(str(EXP/'delivery/stage1'/s/'frames/0000.png')) for s in geom.SIDS}
    pairs = [('FF_02','SL_02', first['FF_02'], first['SL_02'])]
    for s in geom.SIDS:
        count = next(q['exported_frames'] for q in geom.obs['samples'] if q['sample_id']==s)
        pairs.append((s, s+'_last', first[s], cv2.imread(str(EXP/'delivery/stage1'/s/'frames'/f'{count-1:04d}.png'))))
    result = []
    for src, dst, image_a, image_b in pairs:
        ka, da = orb.detectAndCompute(image_a, mask)
        kb, db = orb.detectAndCompute(image_b, mask)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [m for m, n in matches if m.distance < .7*n.distance]
        pa = np.array([ka[m.queryIdx].pt for m in good])
        pb = np.array([kb[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(101)
        M, inlier = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2)
        residuals = np.linalg.norm(pa @ M[:, :2].T + M[:, 2] - pb, axis=1)
        record = {'from': src, 'to': dst, 'matches': len(good), 'inliers': int(inlier.sum()),
                  'similarity': np.vstack([M, [0,0,1]]).tolist(),
                  'median_inlier_error_px': float(np.median(residuals[inlier[:,0]==1])),
                  'mask_definition': 'static wall/advertisement strip; scoreboard and player region excluded'}
        result.append(record)
        if src == 'FF_02' and dst == 'SL_02':
            assert np.max(abs(np.vstack([M,[0,0,1]])-geom.H)) < 1e-7
            image = cv2.drawMatches(image_a, ka, image_b, kb,
                [m for m, keep in zip(good, inlier[:,0]) if keep], None, flags=2)
            cv2.imwrite(str(ALIGN/'background_registration.jpg'), image)
    return result


def reference_record(sid):
    raw = next(r for r in geom.manifest if r['id']==sid)
    p, v, a, tr = geom.refs[sid]
    front_y = 17/12*geom.FT
    tp = (-v[1]-math.sqrt(v[1]**2+2*a[1]*(front_y-p[1])))/a[1]
    at_plate = p+v*tp+.5*a*tp**2
    given_plate = np.array([float(raw['savant']['plate_x']), float(raw['savant']['plate_z'])])*geom.FT
    error = at_plate[[0,2]]-given_plate
    assert abs(p[1]-15.24) < 1e-12 and np.max(abs(error)) < .002
    release_position = np.array([float(raw['savant']['release_pos_'+k]) for k in 'xyz'])*geom.FT
    assert np.max(abs(p+v*tr+.5*a*tr**2-release_position)) < 1e-10
    return {'sample_id': sid, 'play_id': raw['play_id'], 'pitch_type': raw['pitch_type'],
            'split': geom.rows_for(sid,'all')[0]['split'], 'raw_statcast': raw['savant'],
            'state_reference': 't_world=0 when y=50ft=15.24m', 'position_at_reference_m': p.tolist(),
            'velocity_at_reference_mps': v.tolist(), 'net_acceleration_mps2': a.tolist(),
            'non_gravity_acceleration_mps2': (a-np.array([0,0,-9.81])).tolist(),
            'release_time_world_s': float(tr), 'release_position_m': release_position.tolist(),
            'release_velocity_mps': (v+a*tr).tolist(), 'plate_front_y_m': front_y,
            'plate_front_time_world_s': float(tp), 'release_to_plate_s': float(tp-tr),
            'reconstructed_plate_xz_m': at_plate[[0,2]].tolist(),
            'recorded_plate_xz_m': given_plate.tolist(), 'plate_reference_residual_xz_m': error.tolist(),
            'plate_reference_residual_norm_m': float(np.linalg.norm(error)),
            'time_root_choice': 'near physical root; negative t for release, positive t for plate',
            'reference_model': 'constant net acceleration from supplied Statcast fit; not independent 3D truth'}


def main():
    for path in [ALIGN, DELIVERY, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT/'data/manifest.json', ROOT/'data/annotations.json', ROOT/'data/tracknet_labels.json',
                    ROOT/'data/tracknet_ff_sl/dataset.json', EXP/'observations/observations_2d.json']
    source_paths += [Path(next(s['source_video'] for s in geom.obs['samples'] if s['sample_id']==sid)) for sid in geom.SIDS]
    before = {str(p): sha256(p) for p in source_paths}
    assert all(geom.rows_for(s, 'all')[0]['split']=='train' for s in geom.SIDS)
    background = verify_background()
    print('Background registration checked', flush=True)
    refs = {s: reference_record(s) for s in geom.SIDS}
    q, candidates = geom.fit('all')
    q_prefix, prefix_candidates = geom.fit('prefix')
    cameras = {s: camera_record(q,s) for s in geom.SIDS}
    dump(ALIGN/'camera.json', {'schema_version':1, 'status':'effective_local_alignment_estimated',
        'world_coordinates': {'origin':'back tip of home plate', 'x':'catcher right',
                              'y':'toward pitcher', 'z':'up', 'units':'m'},
        'projection_convention':'X_cam=R_world_to_camera @ X_world + T; u right, v down, cam z forward',
        'parameter_assumptions': {'FF_02_principal_point_px':[640,360], 'square_pixels':True,
                                 'distortion_fixed_zero':True, 'shared_effective_center':True,
                                 'SL_image_transform':'measured static-background similarity; encoded in K and R'},
        'parameter_vector':(q*geom.scale).tolist(),
        'parameter_names':['C_x_m','C_y_m','C_z_m','look_at_x_m','look_at_z_m','roll_rad',
                           'log_focal_px','FF_02_video_time_origin_s','SL_02_video_time_origin_s'],
        'cameras':cameras, 'scope_limit':'two train clips, fixed flight windows and nearby spatial region; no hardware or cross-view claim'})
    records, checks = [], []
    for sid in geom.SIDS:
        rows = [r for r in geom.obs['rows'] if r['sample_id']==sid]
        t, X, V, uv = project_rows(q,sid,rows)
        _, _, _, uv_prefix = project_rows(q_prefix,sid,rows)
        origin = float((q*geom.scale)[7+geom.SIDS.index(sid)])
        visible = [i for i,r in enumerate(rows) if r['use_for_observation_loss']]
        prefix_ids = {r['frame_index'] for r in geom.rows_for(sid,'prefix')}
        errors = [np.linalg.norm(uv[i]-[rows[i]['u_px'],rows[i]['v_px']]) for i in visible]
        future = [i for i in visible if rows[i]['frame_index'] not in prefix_ids]
        prefix_future_errors = [np.linalg.norm(uv_prefix[i]-[rows[i]['u_px'],rows[i]['v_px']]) for i in future]
        camera = cameras[sid]
        R = np.array(camera['R_world_to_camera']);T=np.array(camera['T_world_to_camera_m']);K=np.array(camera['K'])
        depth = (X@R.T+T)[:,2]
        # Nominal diameter from 9-9.25 inch circumference; a size estimate only.
        diameter = K[0,0]*.073/depth
        visible_core_ratios=[]
        summary = {'sample_id':sid, 'calibration_error':stats(errors),
                   'prefix_fitted_manual_frames':len(prefix_ids),
                   'prefix_only_later_frame_check':stats(prefix_future_errors),
                   'median_error_over_estimated_ball_diameter':float(np.median(np.array(errors)/diameter[visible])),
                   'estimated_ball_diameter_px_range':[float(min(diameter)),float(max(diameter))],
                   'ball_size_source':'0.073m nominal diameter projected by fitted camera; not measured independent truth',
                   'video_time_origin_s':origin, 'world_time_reference':'y=50ft',
                   'release_video_time_s':origin+refs[sid]['release_time_world_s'],
                   'plate_front_video_time_s':origin+refs[sid]['plate_front_time_world_s'],
                   'prefix_video_time_origin_s':float((q_prefix*geom.scale)[7+geom.SIDS.index(sid)]),
                   'origin_fit_scope':'all manual frames of two train pitches; calibration, not future prediction',
                   'prefix_fit_frame_ids':sorted(prefix_ids)}
        checks.append(summary)
        directory = DELIVERY/sid
        frames = directory/'frames'
        frames.mkdir(parents=True,exist_ok=True)
        for i,row in enumerate(rows):
            image = cv2.imread(str(EXP/'delivery/stage1'/sid/'frames'/f'{i:04d}.png'))
            core=visible_ball_core(image,row)
            error = None
            if row['use_for_observation_loss']:
                error = float(np.linalg.norm(uv[i]-[row['u_px'],row['v_px']]))
                if core:
                    visible_core_ratios.append(error/core['equivalent_diameter_px'])
                cv2.circle(image,(round(row['u_px']),round(row['v_px'])),9,(40,220,40),1,cv2.LINE_AA)
                cv2.line(image,(round(row['u_px']),round(row['v_px'])),tuple(np.rint(uv[i]).astype(int)),(255,255,255),1)
            cv2.drawMarker(image,tuple(np.rint(uv[i]).astype(int)),(0,165,255),cv2.MARKER_CROSS,7,1)
            cv2.rectangle(image,(10,10),(1140,76),(15,15,15),-1)
            caption=f"{sid} frame={row['frame_index']} world_t={t[i]:.6f}s reprojection_error={error if error is not None else 'no observation'}"
            cv2.putText(image,caption,(20,36),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image,'GREEN=reviewed center  ORANGE=Statcast 3D reference projection (not learned prediction)',(20,62),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            assert cv2.imwrite(str(frames/f'{i:04d}.png'),image)
            records.append({'sample_id':sid,'play_id':row['play_id'],'pitch_type':sid[:2],
                'split':row['split'],'frame_index':row['frame_index'],'label_frame_index':row['label_frame_index'],
                'video_pts_s':row['video_pts_s'],'container_pts_s':row['container_pts_s'],'t_world_s':float(t[i]),
                'position_m':X[i].tolist(),'velocity_mps':V[i].tolist(),
                'net_acceleration_mps2':geom.refs[sid][2].tolist(),'state_source':'reference',
                'reference_source':'Statcast constant-net-acceleration parameterization',
                'event_type':'none','visibility':row['visibility'],'observed_u_px':row['u_px'],'observed_v_px':row['v_px'],
                'projected_u_px':float(uv[i,0]),'projected_v_px':float(uv[i,1]),'error_px':error,
                'estimated_ball_diameter_px':float(diameter[i]),'visible_ball_core_measurement':core,
                'error_over_visible_core_diameter':error/core['equivalent_diameter_px'] if core else None,
                'camera_segment_id':camera['camera_segment_id']})
        summary['visible_core_size_check']={'accepted_frames':len(visible_core_ratios),
                'median_error_over_core_diameter':float(np.median(visible_core_ratios)),
                'p90_error_over_core_diameter':float(np.quantile(visible_core_ratios,.9)),
                'not_precise_full_ball_diameter':True}
        source_summary=next(s for s in geom.obs['samples'] if s['sample_id']==sid)
        last_duration=source_summary['clip_duration_s']-source_summary['first_to_last_duration_s']
        encode_images(frames,[r['clip_pts_s'] for r in rows],last_duration,directory/'reprojection_overlay.mp4')
        output=probe(directory/'reprojection_overlay.mp4')
        assert len(output['frames'])==len(rows)
        assert (output['streams'][0]['width'],output['streams'][0]['height'])==(1280,720)
        assert abs(float(output['streams'][0]['duration'])-source_summary['clip_duration_s'])<1/90000
        output_tb=Fraction(output['streams'][0]['time_base'])
        exported_pts=[float(int(f.get('pts',f['best_effort_timestamp']))*output_tb) for f in output['frames']]
        max_pts_error=max(abs(a-b['clip_pts_s']) for a,b in zip(exported_pts,rows))
        assert max_pts_error < 1/90000*1.1
        summary['overlay_checks']={'frame_count':len(rows),'duration_s':float(output['streams'][0]['duration']),
                                   'image_size':[1280,720],'max_pts_error_s':max_pts_error}
        overview=Image.new('RGB',(1280,((len(visible)+7)//8)*155),'white')
        draw=ImageDraw.Draw(overview)
        for n,i in enumerate(visible):
            row=rows[i];x,y=round(row['u_px']),round(row['v_px'])
            patch=Image.open(frames/f'{i:04d}.png').crop((x-24,y-20,x+24,y+20)).resize((144,120))
            left,top=(n%8)*160+8,(n//8)*155+26;overview.paste(patch,(left,top))
            draw.text((left,top-20),f"{sid} frame {row['frame_index']}",fill='black')
        overview.save(directory/'reprojection_contact_sheet.png')
        print(sid,json.dumps(summary['calibration_error']),flush=True)
    dump(ALIGN/'trajectory_reference_3d.json',{'schema_version':1,'units':'SI',
        'coordinate_convention':'x catcher right; y toward pitcher; z up; origin home plate back tip',
        'time_convention':'t_world=video_pts_s-video_time_origin_s; world zero at y=50ft',
        'construction':'infer p50 by back-extrapolating from release using v50 and constant net a',
        'reference_is_not_independent_3d_truth':True,'gravity_is_already_in_net_acceleration':True,
        'raw_source':str(ROOT/'data/manifest.json'),'field_definition_sources':SOURCES,
        'samples':refs,'rows':records})
    fields=['sample_id','frame_index','video_pts_s','t_world_s','observed_u_px','observed_v_px',
            'projected_u_px','projected_v_px','error_px','estimated_ball_diameter_px',
            'error_over_visible_core_diameter','visibility']
    with (ALIGN/'reprojection_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    field_error=np.linalg.norm(geom.project(q,geom.field_world,'FF_02')-geom.field_uv,axis=1)
    field_names=['left_box_outer_rear','left_box_outer_front','left_box_inner_rear','left_box_inner_front',
                 'right_box_inner_rear','right_box_inner_front','right_box_outer_rear','right_box_outer_front','pitching_rubber_center_front']
    dump(ALIGN/'scene_correspondences.json',{'source_frame':'FF_02 stage1 frames/0000.png',
         'annotation_source':'agent visually identified chalk/rubber landmarks; original ball annotations unchanged',
         'geometry_source':SOURCES['field_geometry']+'#page=173','box_center_y_m':8.5/12*geom.FT,
         'box_dimensions_m':[4*geom.FT,6*geom.FT],'plate_width_m':17/12*geom.FT,
         'box_gap_from_plate_m':.5*geom.FT,'mound_height_m':10/12*geom.FT,
         'pixel_uncertainty_assumption':3,'limitation':'approximate chalk centers versus nominal outer box edges; partial occlusion and line thickness limit metric precision',
         'points':[{'name':n,'world_m':x.tolist(),'image_px':u.tolist(),'reprojection_error_px':float(e)}
                   for n,x,u,e in zip(field_names,geom.field_world,geom.field_uv,field_error)]})
    image=cv2.imread(str(EXP/'delivery/stage1/FF_02/frames/0000.png'))
    for i,(target,prediction) in enumerate(zip(geom.field_uv,geom.project(q,geom.field_world,'FF_02'))):
        cv2.circle(image,tuple(np.rint(target).astype(int)),5,(40,220,40),1)
        cv2.drawMarker(image,tuple(np.rint(prediction).astype(int)),(0,165,255),cv2.MARKER_CROSS,7,1)
        cv2.putText(image,str(i),tuple(np.rint(target+[5,-5]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.imwrite(str(ALIGN/'scene_reprojection.png'),image)
    # Profile fixed focal values: geometry identifiability check, not motion model comparison.
    profiles=[]
    for focal in [8000,12000,16000,22000,28000,35000]:
        start=q.copy();ratio=focal/np.exp((q*geom.scale)[6]);start[:3]*=ratio
        qq,cost,it=geom.optimize(start,fixed={6:np.log(focal)})
        errs=[]
        for sid in geom.SIDS:
            rows=geom.rows_for(sid,'all');_,_,_,uv=project_rows(qq,sid,rows)
            errs.extend(np.linalg.norm(uv-np.array([[r['u_px'],r['v_px']] for r in rows]),axis=1).tolist())
        profiles.append({'focal_px':focal,'cost':float(cost),'params':(qq*geom.scale).tolist(),
                         'all_ball_errors':stats(errs),'camera_center_world_m':(qq*geom.scale)[:3].tolist()})
    dump(ALIGN/'parameter_sensitivity.json',{'purpose':'profile focal length and pose ambiguity under fixed reference geometry',
         'not_motion_model_ablation':True,'profiles':profiles,'scope':'two clips and approximate field points'})
    alignment={'status':'completed_effective_local_calibration','samples':checks,'static_background_registration':background,
        'field_point_errors':stats(field_error),'full_fit_cost':float(candidates[0][0]),
        'prefix_fit_cost':float(prefix_candidates[0][0]),'full_fit_multistart_count':len(candidates),
        'multistart_best_worst_cost':[float(candidates[0][0]),float(candidates[-1][0])],
        'prefix_only_camera':{s:camera_record(q_prefix,s,'prefix') for s in geom.SIDS},
        'prefix_only_parameter_vector':(q_prefix*geom.scale).tolist(),
        'prefix_check_interpretation':'camera/time calibration using first third only, checking later reference projections; not Neural ODE prediction',
        'fit_weighting':'ball coordinate residuals /1px; approximate field coordinate residuals /3px',
        'bounds_physical_params':{'lower':(geom.lo*geom.scale).tolist(),'upper':(geom.hi*geom.scale).tolist()},
        'official_hardware_parameters_recovered':False,'source_hashes':before,'sources':SOURCES,
        'next_stage':'camera can be held fixed for current train scene; new clips require scene/PTS registration before motion training',
        'limitations':['reference comes from constant-acceleration Statcast fits',
                       'camera pose/focal estimates depend on approximate chalk geometry and time assumptions',
                       'distortion and within-frame exposure timing not estimated',
                       'full-span fit is calibration evidence, not future-state prediction',
                       'no independent multiview/depth or actual hardware validation']}
    assert before=={str(p):sha256(p) for p in source_paths}
    alignment['source_data_unchanged']=True
    dump(ALIGN/'alignment.json',alignment)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,sid in zip(axes,geom.SIDS):
        rr=[r for r in records if r['sample_id']==sid and r['error_px'] is not None]
        ax.plot([r['video_pts_s'] for r in rr],[r['error_px'] for r in rr],'o-',label='Full-span calibration',ms=3)
        rows=geom.rows_for(sid,'all');_,_,_,up=project_rows(q_prefix,sid,rows)
        er=np.linalg.norm(up-np.array([[r['u_px'],r['v_px']] for r in rows]),axis=1)
        ax.plot([r['video_pts_s'] for r in rows],er,'--',label='Prefix-only camera/time fit')
        ax.axvline(rows[len(geom.rows_for(sid,'prefix'))-1]['video_pts_s'],color='gray',ls=':',label='Prefix boundary')
        ax.set(title=sid,xlabel='Video PTS from first frame (s)',ylabel='Reprojection center error (px)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(ALIGN/'reprojection_error_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot([p['focal_px'] for p in profiles],[p['all_ball_errors']['p90_px'] for p in profiles],'o-')
    axes[0].set(xlabel='Fixed focal length (px)',ylabel='Ball reprojection P90 (px)',title='Geometry parameter sensitivity')
    axes[1].plot([p['focal_px'] for p in profiles],[p['camera_center_world_m'][1] for p in profiles],'o-')
    axes[1].set(xlabel='Fixed focal length (px)',ylabel='Fitted camera Y (m)',title='Focal length / distance coupling')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(ALIGN/'parameter_sensitivity.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for sid in geom.SIDS:
        p,v,a,tr=geom.refs[sid];ts=np.linspace(tr,refs[sid]['plate_front_time_world_s'],200);X=p+v*ts[:,None]+.5*a*ts[:,None]**2
        axes[0].plot(X[:,1],X[:,0],label=sid);axes[1].plot(X[:,1],X[:,2],label=sid)
    axes[0].set(xlabel='Y toward pitcher (m)',ylabel='X catcher right (m)',title='Statcast reference: horizontal')
    axes[1].set(xlabel='Y toward pitcher (m)',ylabel='Z up (m)',title='Statcast reference: vertical')
    for ax in axes:ax.invert_xaxis();ax.grid(alpha=.2);ax.legend()
    fig.savefig(ALIGN/'reference_trajectory_views.png',dpi=170);plt.close(fig)
    report=['# 阶段 2：三维参考、时间对应与有效相机配准','',
        '状态：已完成 FF_02、SL_02 的局部有效相机配准，输出三维参考状态、相机/时间参数、重投影视频及检查结果。',
        '这些结果支持当前镜头附近的三维到二维对应，不代表恢复了电视台相机的真实硬件参数。','',
        '## 三维参考与单位核对','',
        '采用米、秒、米/秒、米/秒²；1 ft = 0.3048 m。世界原点为本垒板后尖，x 向捕手右侧、y 向投手、z 向上。',
        '速度参考面为 y=50 ft；先由释放位置、参考速度和净加速度求负的释放时间，再反推同一时刻的位置 p50。',
        '本批为 2023 年数据，过本垒参考面采用本垒板前沿 y=17/12 ft；不使用 2026 年中部参考面。净加速度已包含重力，另存减去 g 后的非重力加速度，不重复叠加。',
        '下表检查字段/参考面的一致性，不将毫米级差异解释为真实三维测量精度。','',
        '| 样本 | 释放相对 y=50ft 时刻 | 释放到前沿时长 | 重建前沿 x/z 与记录差异 |','|---|---:|---:|---:|']
    for sid in geom.SIDS:
        r=refs[sid];report.append(f"| {sid} | {r['release_time_world_s']:.6f} 秒 | {r['release_to_plate_s']:.6f} 秒 | {r['plate_reference_residual_norm_m']*1000:.3f} 毫米 |")
    report += ['', '## 相机和画面对应','',
        '先识别画面中打击区线和投手板作为近似几何约束，再联合拟合两条训练投球的参考投影和各自时间偏移。场地点由本轮从画面识别，原人工球心未改动。打击区粗线、部分遮挡和线中心/边界差别限制场地点的精度，详见 scene_correspondences.json。',
        f"背景对应保留 {background[0]['inliers']} 个内点，中位残差 {background[0]['median_inlier_error_px']:.3f} 像素；SL 相对 FF 的缩放为 {np.sqrt(np.linalg.det(geom.H[:2,:2])):.6f}。缩放/平移/微小画面旋转编码到各段 K/R，不能因同场比赛而直接共用同一 K。",
        '固定 FF 主点为 (640,360)，采用方形像素、零畸变，估计有效机位、朝向、焦距和两条时间偏移；背景相似变换支持局部共享有效相机中心，不是对实际硬件机位的独立测量。',
        f"近似场地点重投影中位误差 {np.median(field_error):.3f} 像素，最大 {max(field_error):.3f} 像素；场地约束仍有误差，不能称为精确场地重建。",'',
        '## 重投影检查','',
        '| 样本 | 人工点数 | 全段拟合中位误差 | P90 | 最大误差 | 仅前段拟合后的后段 P90 |','|---|---:|---:|---:|---:|---:|']
    for s in checks:
        e=s['calibration_error'];p=s['prefix_only_later_frame_check'];report.append(f"| {s['sample_id']} | {e['count']} | {e['median_px']:.3f} px | {e['p90_px']:.3f} px | {e['max_px']:.3f} px | {p['p90_px']:.3f} px |")
    report += ['', '绿色圆环为原人工球心，橙色十字为三维参考的相机投影；缺失人工观测帧只有参考投影，不伪造绿色观测。',
        '估计球径采用 0.073 m 名义球径和相机深度，误差/估计球径比仅作尺度参考，不是逐帧实际球径测量。',
        '另从原画面提取球心附近的低色差亮连通域，估计可见亮核的面积等效直径；剔除极小碎片、触及局部窗口边界或过度细长的区域。亮核不包含完整模糊轮廓，仍受白色背景和阈值影响，不作为精确球径真值。',
        '全段拟合使用两条 train 投球的全部 53 个人工点，这是配准证据。另用每条前 9 个人工点独立拟合相机/时间，再检查其余帧；这是前段配准的后段一致性检查，还不是 Neural ODE 预测。',
        'video_time_origin_s 对应 y=50 ft，不是释放时刻；释放时间和本垒前沿时间另存。沿用阶段 1 首帧归零的 video_pts_s，原容器 PTS 单独保存。', '',
        '## 参数歧义与后续使用','',
        '通过不同初值核对收敛，并固定若干焦距重拟合其余参数，检查焦距/机位距离耦合。不同焦距下仍可能有较小球心误差，参数不宜解释为真实硬件实测值。参数敏感性是几何配准检查，不是运动模型对照或消融。',
        f"例如焦距固定 12000 px 时，有效相机 y={profiles[1]['camera_center_world_m'][1]:.1f} m，合并球心 P90={profiles[1]['all_ball_errors']['p90_px']:.2f} px；固定 28000 px 时，y={profiles[4]['camera_center_world_m'][1]:.1f} m，P90={profiles[4]['all_ball_errors']['p90_px']:.2f} px。35000 px 的结果触及机位距离上界，仅作受限拟合记录。", 
        '阶段 3 可对当前场景固定这组有效相机，再扩展训练数据；其他片段必须先检查取景、PTS 和背景对应。相机与参考轨迹误差会进入后续监督，应保留来源。单目配准不独立证明深度准确，初速度编辑应限制在已检查空间范围内。', '',
        '## 文件与复现','',
        '- [三维参考状态](../alignment/trajectory_reference_3d.json)',
        '- [相机参数](../alignment/camera.json)', '- [时间对应和检查明细](../alignment/alignment.json)',
        '- [误差曲线](../alignment/reprojection_error_curves.png)',
        '- [参数敏感性](../alignment/parameter_sensitivity.png)',
        '- [视频查看页](../delivery/stage2/index.html)', '',
        '```sh','bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/align_stage2.py','```','',
        f"运行环境：Python {sys.version.split()[0]}，NumPy {np.__version__}，OpenCV {cv2.__version__}，Matplotlib {matplotlib.__version__}。本轮只在本地执行，未训练运动模型。",'',
        '## 字段与几何来源','']
    report += [f'- [{name}]({url})' for name,url in SOURCES.items()]
    report += ['', '## 相对于可见亮核大小的误差','',
               '| 样本 | 可提取亮核的帧数 | 误差/亮核直径中位值 | P90 |',
               '|---|---:|---:|---:|']
    for s in checks:
        core=s['visible_core_size_check'];report.append(f"| {s['sample_id']} | {core['accepted_frames']} | {core['median_error_over_core_diameter']:.3f} | {core['p90_error_over_core_diameter']:.3f} |")
    (REPORTS/'alignment_report.md').write_text('\n'.join(report)+'\n')
    (ALIGN/'alignment_report.md').write_text('\n'.join(report).replace('../alignment/','./').replace('../delivery/','../delivery/')+'\n')
    cards=''.join(f'<section><h2>{s}</h2><video controls loop src="{s}/reprojection_overlay.mp4"></video>'
                  f'<p><a href="{s}/reprojection_contact_sheet.png">逐点重投影检查</a></p></section>' for s in geom.SIDS)
    (DELIVERY/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段 2 重投影</title>'
        '<style>body{font:16px system-ui;max-width:1280px;margin:32px auto;background:#f4f5f7;padding:24px}section{background:white;padding:20px;margin:24px 0}video{width:100%}</style>'
        '<h1>三维参考与转播画面配准</h1><p>绿色：已检查球心；橙色：Statcast 三维参考投影。尚未训练运动模型。</p>'+cards+'</html>')
    config_path=EXP/'config.json';config=json.loads(config_path.read_text());config['stage2_status']=alignment['status'];
    config['stage2_reproduce_command']='bradish_pilot/.venv/bin/python bradish_pilot/experiments/neural_ode_ball/align_stage2.py';dump(config_path,config)
    print('Stage 2 complete:',REPORTS/'alignment_report.md',flush=True)


if __name__=='__main__':
    main()
