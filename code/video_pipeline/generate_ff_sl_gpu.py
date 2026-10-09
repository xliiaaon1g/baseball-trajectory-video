"""Paired real-first-frame FF/SL-shape Wan-Move GPU trial.

Consumes the geometric preview's conditions. This generates two new videos
from the SAME first frame, prompt, seed, resolution and inference settings;
it is not a pixel-preserving editor of the source video. Run only with an
installed, pinned Wan-Move checkout and complete official weights.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
INPUT = ROOT / 'data/tracknet_ff_sl/ff_to_sl_model_preview'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    import numpy as np
    import torch
    from PIL import Image
    from ball_control_experiment import validate_inputs, export_blind
    pinned = json.loads((PROJECT / 'pilot_pipeline/weights_manifest.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    issues = []
    missing_imports = [name for name in ['decord','diffusers','transformers','easydict','flash_attn','torchvision']
                       if importlib.util.find_spec(name) is None]
    if missing_imports:
        issues.append('Model runtime dependencies missing')
    if tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2]) < (2,4):
        issues.append('Wan-Move requires torch >=2.4')
    if not torch.cuda.is_available():
        issues.append('CUDA unavailable; this generator requires NVIDIA GPU')
    else:
        if torch.cuda.get_device_properties(0).total_memory / 2**30 < 39:
            issues.append('Official bf16 route requires single 40GB class or larger')
        if not torch.cuda.is_bf16_supported():
            issues.append('bf16 unavailable')
    try:
        commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != pinned['model_commit']:
            issues.append('Wan-Move commit differs from pinned project interface')
        if subprocess.check_output(['git', '-C', str(args.repo), 'status', '--porcelain',
                                    '--untracked-files=no'], text=True).strip():
            issues.append('Tracked Wan-Move source has local changes')
    except (OSError, subprocess.CalledProcessError):
        issues.append('Wan-Move checkout missing')
    missing = [q['path'] for q in pinned['required_files'] if not (args.ckpt/q['path']).is_file()
               or (args.ckpt/q['path']).stat().st_size != q['bytes']]
    if missing:
        issues.append('Official weights missing or wrong size')
    if shutil.disk_usage(args.out).free < 2 * 10**9:
        issues.append('Less than 2GB free output space')
    contract = None
    try:
        contract = validate_inputs(args.inputs)
    except (ValueError,OSError,KeyError) as exc:
        issues.append('Input contract failed: '+str(exc))
    first_hash = contract['first_frame_sha256'] if contract else None
    report = {'issues': issues, 'missing_weights': missing, 'missing_imports': missing_imports,
              'status': 'blocked_before_generation' if issues else 'ready_for_gpu_attempt',
              'first_frame_sha256': first_hash, 'input_contract':contract, 'seed': 101,
              'constraints': 'one ball + six stationary background anchors; slow-motion; preservation unverified'}
    dump(args.out / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.run:
        return
    if issues:
        raise RuntimeError('Resolve preflight issues first')
    for name in ['original_ff','sl_shape_candidate']:
        if (args.out/name).exists():
            raise RuntimeError('Refuse overwrite: '+str(args.out/name))
    # Reuse project's complete official-weight published SHA256 verifier.
    sys.path.insert(0, str(PROJECT / 'pilot_pipeline'))
    from run_gpu import verify_weight_hashes
    verify_weight_hashes(args)
    sys.path.insert(0, str(args.repo.resolve()))
    import wan
    from wan.configs import WAN_CONFIGS
    import copy
    config = copy.deepcopy(WAN_CONFIGS['wan-move-i2v'])
    config.param_dtype = torch.bfloat16
    model = wan.WanMove(config=config, checkpoint_dir=str(args.ckpt.resolve()),
                        device_id=0, rank=0, t5_cpu=True, init_on_cpu=True)
    prompt = ('A fixed center-field television broadcast camera shows a baseball pitcher '
              'and catcher in the same stadium scene. A single white baseball travels '
              'slowly through the air toward the catcher following the supplied trajectory. '
              'The camera and stadium background remain steady. The ball stays visible.')
    records = []
    for name in ['original_ff', 'sl_shape_candidate']:
        case, output = args.inputs / name, args.out / name
        if output.exists():
            raise RuntimeError('Refuse overwrite: ' + str(output))
        output.mkdir()
        # Upstream trajectory features use global torch.randperm in addition
        # to the diffusion sampler's seed; reset both for the paired trial.
        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        record = {'case': name, 'status': 'started', 'seed': 101, 'steps': 50,
                  'pairing': 'reset Python, NumPy, torch global and sampler seeds; bitwise equality not asserted',
                  'prompt': prompt, 'first_frame_sha256': first_hash,
                  'tracks_sha256': sha(case / 'tracks.npy')}
        dump(output / 'run.json', record)
        try:
            video = model.generate(prompt, Image.open(case / 'first_frame.png').convert('RGB'),
                                   np.load(case/'tracks.npy').copy(), np.load(case/'visibility.npy').copy(),
                                   max_area=832*480, frame_num=81, shift=3., sample_solver='unipc',
                                   sampling_steps=50, guide_scale=5., seed=101,
                                   offload_model=True, eval_bench=False)
            torch.cuda.synchronize()
            if list(video.shape) != [3, 81, 480, 832]:
                raise RuntimeError('Unexpected output shape')
            rgb = ((video.detach().float().cpu().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,3,0).numpy()
            frames = output / 'frames'
            frames.mkdir()
            for i, frame in enumerate(rgb):
                Image.fromarray(frame).save(frames / f'{i:03d}.png')
            for fname, codec in [('generated.mp4', ['-c:v','libx264','-crf','18','-pix_fmt','yuv420p']),
                                 ('generated_lossless.mkv', ['-c:v','ffv1'])]:
                subprocess.run(['ffmpeg','-v','error','-framerate','16','-i',str(frames/'%03d.png'),
                                *codec,str(output/fname)], check=True)
            record.update(status='generated_manual_review_pending', elapsed_s=time.perf_counter()-started,
                          peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            del video, rgb
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
            dump(output/'run.json', record)
            raise
        dump(output/'run.json', record)
        records.append({'output_case':name,'condition':name,'backend':'Wan-Move',
                        'condition_sha256':record['tracks_sha256']})
    export_blind(args.out,records,args.out/'blind_frames')
    print('Pair complete. Independently annotate blind_frames; keep mapping from annotator.',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT/'data/tracknet_ff_sl/gpu_pair')
    p.add_argument('--inputs', type=Path, default=INPUT)
    p.add_argument('--run', action='store_true')
    main(p.parse_args())
