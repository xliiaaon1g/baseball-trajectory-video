"""Create conservative timing proposals from Statcast kinematics and video cues.

The estimates are review aids, not labels: MLB highlight cuts and broadcast
graphics vary. Confirm both times in review.html before exporting clips.
"""
import json, pathlib, subprocess
import cv2
import numpy as np


def _flight_time(savant):
    """Solve Statcast y(t) for the plate front edge (17 inches from its back point)."""
    try:
        y0=float(savant['release_pos_y']); vy=float(savant['vy0']); ay=float(savant['ay'])
        target=17.0/12.0
        roots=np.roots([0.5*ay,vy,y0-target])
        valid=[float(r.real) for r in roots if abs(r.imag)<1e-7 and 0<float(r.real)<1.0]
        return min(valid) if valid else None
    except (KeyError,ValueError,TypeError):
        return None


def analyze_video(video, row, outdir):
    video=pathlib.Path(video); meta=json.loads(subprocess.check_output([
        'ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)]))
    duration=float(meta['format']['duration']); flight=_flight_time(row['savant'])
    # Circle onset is deliberately manual because field/stand graphics caused
    # false positives in the initial color-based detector.
    ring=None
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 60
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1,y1,x2,y2=int(w*.32),int(h*.38),int(w*.57),int(h*.88)
    step=max(1,round(fps/12)); prev=None; scores=[]; times=[]; i=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if i%step==0:
            gray=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(96,144))
            if prev is not None:
                flow=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,2,15,2,5,1.1,0)
                scores.append(float(np.percentile(cv2.magnitude(flow[...,0],flow[...,1]),85)))
                times.append(i/fps)
            prev=gray
        i+=1
    cap.release()
    release=None
    if len(scores)>=8:
        margin=6; lo=margin; hi=max(lo+1,len(scores)-margin)
        j=lo+int(np.argmax(scores[lo:hi])); release=round(times[j],4)
    plate=None if release is None or flight is None else round(release+flight,4)
    return {'duration':duration,'savant_flight_seconds':flight,
        'circle_first_candidate_seconds':ring,'motion_peak_candidate_seconds':release,
        'plate_time':plate,'release_time':release,
        'method':'pitcher-region optical-flow pulse proposes release; Statcast y(t) flight-time solve proposes plate crossing; both require frame-by-frame manual correction',
        'confidence':'low','review_required':True,
        'warnings':['Optical-flow peak can reflect windup/body motion rather than ball separation; confirm release frame visually.',
                    'Plate time is propagated from the release proposal, not detected from circle onset. Look backward from the circle and ball shadow, then correct it.',
                    'Check whether the pitch is continuous and both events fall within the clip.']}
