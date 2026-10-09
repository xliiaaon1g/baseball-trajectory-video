"""One bounded AutoDL experiment, with durable state and failure shutdown.

Run inside the dedicated instance only. This wrapper does not change model
precision or sampling parameters. The platform shutdown timer is a second cap.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path('/root/autodl-tmp/ffsl_job')
REPO = Path('/root/autodl-tmp/Wan-Move')
CKPT = Path('/root/autodl-tmp/Wan-Move-14B-480P')
STATUS = ROOT / 'job_status.json'
OUT = ROOT / 'gpu_pair'
DEADLINE = 1791204300  # 2026-10-05 12:45 UTC, before platform 13:00 shutdown.
state = {'instance': 'b00044a773-d7f16dca', 'budget_CNY': 30,
         'started_utc_epoch': time.time(), 'deadline_utc_epoch': DEADLINE}


def update(phase, **extra):
    state.update(phase=phase, updated_utc_epoch=time.time(), **extra)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)
    print(phase, flush=True)


def run(phase, command, limit=7200):
    update(phase)
    remaining = min(limit, int(DEADLINE - time.time()))
    if remaining <= 0:
        raise TimeoutError('Experiment deadline reached')
    with (ROOT / (phase + '.log')).open('ab', buffering=0) as log:
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=remaining)


def package():
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and not p.name.endswith('.pyc')
             and p.name != 'delivery_manifest.json']
    entries = []
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        entries.append({'path': str(p.relative_to(ROOT)),
                        'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
    manifest = ROOT / 'delivery_manifest.json'
    manifest.write_text(json.dumps(entries, indent=2))
    archive = Path('/root/ffsl_delivery.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files + [manifest]:
            z.write(p, str(p.relative_to(ROOT)))
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    Path('/root/ffsl_delivery.sha256').write_text(h.hexdigest() + '\n')


def main():
    # Guard against accidentally deploying this wrapper to another project.
    if 'b00044a773-d7f16dca' not in os.uname().nodename:
        raise RuntimeError('Dedicated experiment instance identity mismatch')
    os.environ.update(MAX_JOBS='2', HF_HUB_DISABLE_XET='1',
                      PIP_CACHE_DIR='/root/autodl-tmp/ffsl_pip_cache',
                      HF_HOME='/root/autodl-tmp/ffsl_hf_cache')
    ROOT.mkdir(exist_ok=True)
    try:
        (ROOT / 'constraints.txt').write_text(
            'torch==2.5.1\ntorchvision==0.20.1\nnumpy==1.26.4\n'
            'opencv-python==4.10.0.84\ndiffusers==0.31.0\n'
            'transformers==4.49.0\nflash_attn==2.7.4.post1\n'
            'gradio==5.23.3\nhuggingface_hub==0.29.3\n'
            'scipy==1.15.2\nuvicorn==0.34.0\nfastapi==0.115.12\n')
        run('install_build_tools', [sys.executable, '-m', 'pip', 'install',
                                   'setuptools', 'wheel', 'packaging', 'ninja'])
        run('install_model_dependencies', [sys.executable, '-m', 'pip', 'install',
            '--no-build-isolation', '-r', str(REPO / 'requirements.txt'),
            '-c', str(ROOT / 'constraints.txt'), 'decord', 'huggingface_hub'], 3600)
        with (ROOT / 'environment.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        run('framework_tests', [sys.executable, '-m', 'unittest', 'discover',
                               '-s', 'bradish_pilot/tests', '-v'], 180)
        run('download_weights', [sys.executable, 'pilot_pipeline/run_gpu.py',
                                '--download', '--ckpt', str(CKPT)], 7200)
        run('generate_pair', [sys.executable, 'bradish_pilot/generate_ff_sl_gpu.py',
                             '--repo', str(REPO), '--ckpt', str(CKPT),
                             '--out', str(OUT), '--run'], 18000)
        update('complete_packaging', result='pair_generated_manual_review_pending')
        package()
        update('complete_waiting_backup', delivery='/root/ffsl_delivery.zip')
        # The 30-minute monitor verifies local hashes, then creates this ACK.
        # If the client is offline, keep durable results and stop GPU charges.
        until = min(DEADLINE, time.time() + 1800)
        while time.time() < until and not (ROOT / 'LOCAL_BACKUP_VERIFIED').exists():
            time.sleep(10)
        update('shutdown_requested', result='pair_generated_manual_review_pending',
               local_backup_ack=(ROOT / 'LOCAL_BACKUP_VERIFIED').exists())
    except BaseException:
        update('failed', error=traceback.format_exc())
        try:
            package()
        except Exception:
            (ROOT / 'packaging_error.log').write_text(traceback.format_exc())
        update('shutdown_requested', result='failed')
    finally:
        os.sync()
        subprocess.run(['/usr/bin/shutdown'], check=True)


if __name__ == '__main__':
    main()
