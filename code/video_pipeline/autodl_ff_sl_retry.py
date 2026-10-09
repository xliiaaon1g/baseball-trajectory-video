"""Separate CPU preparation from GPU inference; fixed FF/SL model parameters.
Requires a verified platform timer before launch. Failure, success and deadline
all stop the dedicated instance, regardless of client connectivity.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

INSTANCE = 'b00044a773-d7f16dca'
ROOT = Path('/root/autodl-tmp/ffsl_retry2')
REPO = Path('/root/autodl-tmp/Wan-Move')
CKPT = Path('/root/autodl-tmp/Wan-Move-14B-480P')
SHUTDOWN = ['/bin/bash', '/usr/bin/shutdown']
WHEEL = ('https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/'
         'flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl')
state = {}
deadline = 0


def update(phase, **fields):
    state.update(phase=phase, updated_utc_epoch=time.time(), **fields)
    tmp = ROOT / 'job_status.tmp'
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(ROOT / 'job_status.json')
    print(phase, flush=True)


def shutdown():
    os.sync()
    with (ROOT / 'shutdown.log').open('ab', buffering=0) as log:
        # AutoDL provides a shell wrapper without a valid executable header.
        # Using bash is verified by a real failure/shutdown acceptance test.
        subprocess.run(SHUTDOWN, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=45)


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(phase, command, limit=1800):
    update(phase)
    remaining = min(limit, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError('Round deadline reached')
    with (ROOT / (phase + '.log')).open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            if code:
                stop_group(process)
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            # pip can leave compiler children behind; stop its entire group.
            stop_group(process)
            raise


def package():
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts
             and p.name not in {'delivery_manifest.json', 'run.lock'}
             and p.suffix not in {'.pyc', '.whl'}]
    manifest = ROOT / 'delivery_manifest.json'
    entries = []
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        entries.append(dict(path=str(p.relative_to(ROOT)),
                            bytes=p.stat().st_size, sha256=h.hexdigest()))
    manifest.write_text(json.dumps(entries, indent=2))
    archive = Path('/root/ffsl_retry2_delivery.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files + [manifest]:
            z.write(p, str(p.relative_to(ROOT)))
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    archive.with_suffix('.sha256').write_text(h.hexdigest() + '\n')


def receive_wheel():
    # Public official release, split locally to avoid a slow single SSH stream.
    cache = Path('/root/autodl-tmp/ffsl_binary_cache')
    entries = json.loads((cache/'parts.json').read_text())
    while time.time() < deadline:
        if all((cache/e['name']).is_file() and (cache/e['name']).stat().st_size == e['bytes'] for e in entries):
            target = cache/'flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
            pending = cache/'verified-wheel.tmp'
            full = hashlib.sha256()
            with pending.open('wb') as out:
                for entry in entries:
                    h = hashlib.sha256()
                    with (cache/entry['name']).open('rb') as inp:
                        for chunk in iter(lambda:inp.read(8*1024*1024),b''):
                            h.update(chunk); full.update(chunk); out.write(chunk)
                    assert h.hexdigest() == entry['sha256'], 'Transferred chunk corrupted'
            assert full.hexdigest() == 'a496f383c843ed0cc6e01302556a0947f56828f1ec2b54e5297c4c7eaae39357', 'Transferred official binary corrupted'
            pending.replace(target)
            print('Official wheel verified and cached', flush=True)
            return
        print('Awaiting official binary chunks', [(e['name'],(cache/e['name']).stat().st_size if (cache/e['name']).exists() else 0) for e in entries], flush=True)
        time.sleep(20)
    raise TimeoutError('Official wheel transfer deadline reached')


def prepare():
    # Install only compatible published binaries. No source compilation fallback.
    run('wheel_environment', [sys.executable, '-c',
        "import torch,sys; assert sys.version_info[:2]==(3,12); "
        "assert torch.__version__.startswith('2.5.1'); "
        "assert torch.version.cuda.startswith('12.'); "
        "assert not torch._C._GLIBCXX_USE_CXX11_ABI; "
        "print(sys.version,torch.__version__,torch.version.cuda)"])
    name = 'flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
    cache = Path('/root/autodl-tmp/ffsl_binary_cache')
    cache.mkdir(exist_ok=True)
    wheel = cache / name
    if not wheel.is_file() or wheel.stat().st_size != 187814472:
        run('receive_flash_wheel', [sys.executable, __file__, 'receive-wheel',
            '--deadline-utc', str(deadline)], 5400)
    h = hashlib.sha256()
    with wheel.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    assert h.hexdigest() == 'a496f383c843ed0cc6e01302556a0947f56828f1ec2b54e5297c4c7eaae39357', 'Official binary differs from verified release download'
    (ROOT/'flash_wheel_provenance.json').write_text(json.dumps(dict(url=WHEEL,bytes=wheel.stat().st_size,sha256=h.hexdigest()),indent=2))
    run('install_flash_wheel', [sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheel)], 180)
    requirements = []
    for line in (REPO / 'requirements.txt').read_text().splitlines():
        # The batch adapter does not import the interactive Gradio web app.
        if line.strip() and not line.lstrip().startswith(('flash_attn', 'gradio')):
            requirements.append(line)
    (ROOT / 'requirements-runtime.txt').write_text('\n'.join(requirements) + '\ndecord\nhuggingface_hub==0.29.3\n')
    (ROOT / 'constraints.txt').write_text(
        'torch==2.5.1\ntorchvision==0.20.1\nnumpy==1.26.4\n'
        'opencv-python==4.10.0.84\ndiffusers==0.31.0\n'
        'transformers==4.49.0\nflash_attn==2.7.4.post1\n'
        'huggingface_hub==0.29.3\nscipy==1.15.2\n')
    run('install_runtime', [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
        '-r', str(ROOT / 'requirements-runtime.txt'), '-c', str(ROOT / 'constraints.txt')], 1800)
    run('validate_runtime', [sys.executable, '-c',
        "import torch,flash_attn,flash_attn_2_cuda,decord,diffusers,transformers,easydict,torchvision; "
        "import compileall; "
        "assert compileall.compile_dir('/root/autodl-tmp/Wan-Move/wan',quiet=1); "
        "print('CPU dependency imports and Wan-Move syntax OK; upstream CUDA initialization deferred to GPU acceptance')"], 180)
    run('framework_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'bradish_pilot/tests', '-v'], 180)
    run('download_weights', [sys.executable, 'pilot_pipeline/run_gpu.py', '--download', '--ckpt', str(CKPT)], 18000)
    run('verify_weights', [sys.executable, '-c',
        "import sys; from pathlib import Path; from types import SimpleNamespace; "
        "sys.path.insert(0,'pilot_pipeline'); from run_gpu import verify_weight_hashes; "
        "verify_weight_hashes(SimpleNamespace(ckpt=Path('/root/autodl-tmp/Wan-Move-14B-480P'),out=Path('.')))"] , 1800)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'environment.txt').write_text(freeze)
    (ROOT / 'PREPARATION_VERIFIED.json').write_text(json.dumps(dict(
        instance=INSTANCE, verified_utc_epoch=time.time(),
        upstream_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        weights_manifest_sha256=hashlib.sha256((ROOT/'pilot_pipeline/weights_manifest.json').read_bytes()).hexdigest(),
        shutdown_probe_verified=True),indent=2))
    update('preparation_complete', result='ready_for_gpu_after_platform_timer')


def generate():
    ready = json.loads((ROOT / 'PREPARATION_VERIFIED.json').read_text())
    assert ready['instance'] == INSTANCE and ready['shutdown_probe_verified']
    assert ready['upstream_commit'] == '80c58a7d2ad175fa82a4d57f79f2a1415317dcfa'
    assert (ROOT / 'verified_weight_hashes.json').is_file()
    # Import/cuda validation is repeated on the actual GPU before loading weights.
    run('gpu_flash_acceptance', [sys.executable, '-c',
        "import torch; from flash_attn import flash_attn_func; "
        "q=torch.randn(1,16,2,64,device='cuda',dtype=torch.bfloat16); "
        "o=flash_attn_func(q,q,q); torch.cuda.synchronize(); "
        "assert o.shape==q.shape and torch.isfinite(o).all(); print('BF16 CUDA flash attention OK')"], 180)
    run('gpu_wan_acceptance', [sys.executable, '-c',
        "import sys; sys.path.insert(0,'/root/autodl-tmp/Wan-Move'); import wan; "
        "from wan.configs import WAN_CONFIGS; "
        "assert WAN_CONFIGS['wan-move-i2v'].vae_stride==(4,8,8); print('Actual Wan-Move CUDA initialization OK')"], 180)
    run('generate_pair', [sys.executable, 'bradish_pilot/generate_ff_sl_gpu.py',
        '--repo',str(REPO),'--ckpt',str(CKPT),'--out',str(ROOT/'gpu_pair'),'--run'], 26400)
    update('generation_complete', result='pair_generated_manual_review_pending')


def main():
    global deadline
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare','generate','watchdog','shutdown-test','receive-wheel'])
    p.add_argument('--deadline-utc', type=float, required=True)
    args = p.parse_args()
    assert INSTANCE in os.uname().nodename, 'Dedicated instance only'
    ROOT.mkdir(exist_ok=True)
    deadline = args.deadline_utc
    if args.mode == 'receive-wheel':
        receive_wheel()
        return
    if args.mode == 'watchdog':
        while time.time() < deadline:
            time.sleep(min(10, max(0, deadline-time.time())))
        (ROOT / 'watchdog_deadline.txt').write_text(str(time.time()))
        shutdown()
        return
    assert 0 < deadline-time.time() <= 8*3600, 'Explicit bounded deadline required'
    lock = (ROOT/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.mode != 'shutdown-test':
        assert (ROOT/'SHUTDOWN_PROBE_UI_VERIFIED.json').is_file(), 'Verify actual shutdown first'
    os.environ.update(HF_ENDPOINT='https://hf-mirror.com',HF_HUB_DISABLE_XET='1',
                      PIP_CACHE_DIR='/root/autodl-tmp/ffsl_pip_cache',
                      HF_HOME='/root/autodl-tmp/ffsl_hf_cache',OMP_NUM_THREADS='1')
    state.update(instance=INSTANCE, round='retry2', budget_CNY=25, mode=args.mode,
                 started_utc_epoch=time.time(), deadline_utc_epoch=deadline)
    subprocess.Popen([sys.executable,__file__,'watchdog','--deadline-utc',str(deadline+60)],
                     stdout=(ROOT/'watchdog.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    try:
        if args.mode == 'shutdown-test':
            raise RuntimeError('Intentional failure for shutdown acceptance')
        if args.mode == 'prepare': prepare()
        else: generate()
    except BaseException:
        update('failed', result='failed', error=traceback.format_exc())
    finally:
        try:
            package()
        except BaseException:
            (ROOT/'packaging_error.log').write_text(traceback.format_exc())
        # Do not keep GPU running while waiting for a desktop backup.
        update('shutdown_requested', previous_result=state.get('result'), delivery='/root/ffsl_retry2_delivery.zip')
        shutdown()


if __name__ == '__main__':
    main()
