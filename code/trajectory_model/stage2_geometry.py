import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
obs=json.loads((EXP/'observations/observations_2d.json').read_text())
manifest=json.loads((ROOT/'data/manifest.json').read_text())
SIDS=['FF_02','SL_02']
FT=.3048

def reference(sid):
 d=next(r for r in manifest if r['id']==sid)['savant']
 rel=np.array([float(d['release_pos_'+k]) for k in 'xyz']);v=np.array([float(d[k]) for k in ['vx0','vy0','vz0']]);a=np.array([float(d[k]) for k in ['ax','ay','az']])
 tr=(-v[1]-np.sqrt(v[1]**2+2*a[1]*(rel[1]-50)))/a[1]
 p=rel-v*tr-.5*a*tr*tr;p[1]=50
 return p*FT,v*FT,a*FT,tr
refs={sid:reference(sid) for sid in SIDS}

def background():
 cv2.setRNGSeed(101)
 a=cv2.imread(str(EXP/'delivery/stage1/FF_02/frames/0000.png'));b=cv2.imread(str(EXP/'delivery/stage1/SL_02/frames/0000.png'));mask=np.zeros(a.shape[:2],np.uint8);mask[160:285,20:1250]=255;mask[180:285,500:820]=0
 orb=cv2.ORB_create(nfeatures=2500);ka,da=orb.detectAndCompute(a,mask);kb,db=orb.detectAndCompute(b,mask);pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for m,n in pairs if m.distance<.7*n.distance];pa=np.array([ka[m.queryIdx].pt for m in good]);pb=np.array([kb[m.trainIdx].pt for m in good]);M,ins=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2)
 return np.vstack([M,[0,0,1]])
H=background()
# Approximate chalk line centers, with 3px uncertainty, not ball labels.
field_world=np.array([[x*FT,y*FT,0] for x,y in [(5.208333,-2.291667),(5.208333,3.708333),(1.208333,-2.291667),(1.208333,3.708333),(-1.208333,-2.291667),(-1.208333,3.708333),(-5.208333,-2.291667),(-5.208333,3.708333)]]+[[0,60.5*FT,10/12*FT]])
field_uv=np.array([[445,407],[437,422],[594,405],[585,422],[703,405],[695,422],[847,405],[843,422],[553,597]],float)
scale=np.array([10,100,10,1,1,.02,1,.01,.01])
lo=np.array([-40,40,2,-2,-2,-.08,np.log(2000),3.0,3.0])/scale
hi=np.array([40,250,30,2,3,.08,np.log(40000),3.2,3.2])/scale

def camera(q):
 p=q*scale;C=p[:3];target=np.array([p[3],0,p[4]]);forward=(target-C);forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.stack([right,down,forward]);co,si=np.cos(p[5]),np.sin(p[5]);R=np.array([[co,-si,0],[si,co,0],[0,0,1]])@R;f=np.exp(p[6]);K=np.array([[f,0,640],[0,f,360],[0,0,1.]])
 return K,R,-R@C

def project(q,X,sid):
 K,R,T=camera(q);cam=X@R.T+T;uv=(cam[:,:2]/cam[:,2:])*K[0,0]+[640,360]
 if sid=='SL_02':uv=uv@H[:2,:2].T+H[:2,2]
 return uv

def rows_for(sid,mode):
 rows=[r for r in obs['rows'] if r['sample_id']==sid and r['use_for_observation_loss']]
 if mode=='prefix':return rows[:(len(rows)+2)//3]
 return rows

def residual(q,mode='all'):
 p=q*scale;out=[]
 for j,sid in enumerate(SIDS):
  rows=rows_for(sid,mode);t=np.array([r['video_pts_s'] for r in rows])-p[7+j];p0,v,a,tr=refs[sid];X=p0+v*t[:,None]+.5*a*t[:,None]**2;uv=project(q,X,sid);target=np.array([[r['u_px'],r['v_px']] for r in rows]);out.extend((uv-target).ravel())
 out.extend(((project(q,field_world,'FF_02')-field_uv)/3).ravel())
 return np.array(out)

def optimize(q,mode='all',iterations=180,fixed=None):
 fixed=fixed or {};active=[j for j in range(len(q)) if j not in fixed]
 q=np.clip(q,lo,hi)
 for j,val in fixed.items():q[j]=val
 lam=.01;fun=lambda q:residual(q,mode);r=fun(q);cost=r@r
 for it in range(iterations):
  eps=1e-5;J=np.column_stack([(fun(q+np.eye(len(q))[j]*eps)-fun(q-np.eye(len(q))[j]*eps))/(2*eps) for j in active]);A=J.T@J;g=J.T@r
  step=np.zeros(len(q));step[active]=np.linalg.lstsq(A+lam*np.diag(np.maximum(np.diag(A),1e-6)), -g,rcond=None)[0]
  candidate=np.clip(q+step,lo,hi);rr=fun(candidate);cc=rr@rr
  if cc<cost:
   change=cost-cc;q=candidate;r=rr;cost=cc;lam=max(1e-10,lam*.4)
   if change<1e-9:break
  else:lam=min(1e10,lam*5)
  if lam>=1e9:break
 return q,cost,it

def fit(mode='all'):
 candidates=[]
 for distance in [60,90,140,200]:
  for timing in [0,.02,-.02]:
   p=np.array([-5*distance/90,distance,7*distance/90,0,.45,0,np.log(110*distance),3.0856778-refs['FF_02'][3]+timing,3.1009333-refs['SL_02'][3]+timing]);q,cost,it=optimize(p/scale,mode);candidates.append((cost,q,it))
 candidates.sort(key=lambda x:x[0]);cost,q,it=candidates[0]
 return q,candidates
