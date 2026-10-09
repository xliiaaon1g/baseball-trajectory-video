"""Reproduce a recorded rollout from state/type/times only; no raw video required."""
from pathlib import Path
import argparse
import json
import sys
import torch

REPO = Path(__file__).resolve().parents[2]
EXP = REPO/'code/trajectory_model'
MODEL = REPO/'models/ball_ode/model.pt'
sys.path.insert(0,str(EXP))
from ode_model import BallODE, integrate


def main(args):
    inp=json.loads(args.input.read_text())
    ckpt=torch.load(MODEL,map_location='cpu',weights_only=False)
    model=BallODE(ckpt['normalization']);model.load_state_dict(ckpt['state_dict']);model.eval()
    encoding={'FF':[1.,0.],'SL':[0.,1.]}[inp['pitch_type']]
    torch.set_num_threads(1)
    with torch.no_grad():
        states=integrate(model,torch.tensor([inp['initial_state']],dtype=torch.float64),
            torch.tensor([encoding],dtype=torch.float64),
            torch.tensor([inp['elapsed_times_s']],dtype=torch.float64))[0]
    result={'input_scope':'initial 3D state + pitch type + timestamps only',
            'elapsed_times_s':inp['elapsed_times_s'],'states':states.tolist()}
    recorded=args.input.with_name(args.input.name.replace('_input.json','_prediction.json'))
    if recorded.exists():
        rows=json.loads(recorded.read_text())['rows']
        expected=torch.tensor([r['position_m']+r['velocity_mps'] for r in rows],dtype=torch.float64)
        result['max_saved_state_difference']=float((states-expected).abs().max())
        assert result['max_saved_state_difference'] < 1e-10
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'frames':len(states),'max_saved_state_difference':result.get('max_saved_state_difference'),
                      'output':str(args.output) if args.output else None},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=REPO/'examples/FF_03_input.json')
    parser.add_argument('--output',type=Path)
    main(parser.parse_args())
