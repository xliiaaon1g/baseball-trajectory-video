"""Extract a TrackNet path without reading any per-frame ball labels.

The time window is supplied separately. This visible-ball-only model always
returns an argmax; it cannot decide whether the ball is absent or occluded.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tracknet_preflight import make_model, ROOT, OUT


def main(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    roi = checkpoint['config']['roi']
    model = make_model(heatmap=True).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open input video')
    context, rows = [], []
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if t > args.end:
                break
            if frame.shape[:2] != (720, 1280):
                raise RuntimeError('Model expects original 1280x720 camera coordinates')
            l, top, w, h = [roi[k] for k in ['left','top','width','height']]
            crop = frame[top:top+h,l:l+w]
            if crop.shape[:2] != (h,w):
                raise RuntimeError('Input incompatible with fixed training ROI')
            context.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
            context = context[-3:]
            if t < args.start or len(context) < 3:
                continue
            x = np.stack(context[::-1]).transpose(0,3,1,2).reshape(9,h,w).copy()
            logits = model(torch.from_numpy(x)[None].float().div(255).to(device))[0,0].cpu().numpy()
            y, x = np.unravel_index(logits.argmax(), logits.shape)
            rows.append({'time_seconds': t,'x': float(x+l),'y': float(y+top),
                         'source_video': str(args.video.resolve()),
                         'label_source': 'tracknet_prediction'})
    cap.release()
    if len(rows) < 3:
        raise RuntimeError('Insufficient flight frames')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'rows':rows,'selected_epoch':checkpoint['epoch'],
                                   'roi':roi,'source':'TrackNet continuous heatmap adaptation',
                                   'per_frame_labels_read':False,
                                   'visibility':'not estimated; forced location on each frame'},indent=2))
    print('Predicted',len(rows),'frames ->',args.out,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--end',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='mps')
    p.add_argument('--checkpoint',type=Path,default=OUT/'run_001/best.pt')
    main(p.parse_args())
