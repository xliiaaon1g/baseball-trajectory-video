"""Portable stage5 runner. Default preflight does not download or run models."""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stage5_media import build_conditions,dump,export_outputs

ROOT=Path(__file__).resolve().parent


def main(args):
    job=json.loads((ROOT/'job.json').read_text());out=ROOT/'outputs';out.mkdir(exist_ok=True)
    missing=[n for n in ['torch','diffusers','transformers','accelerate','cv2'] if importlib.util.find_spec(n) is None]
    cuda=False
    if importlib.util.find_spec('torch'):
        import torch
        cuda=torch.cuda.is_available()
    preflight={'python':platform.python_version(),'cuda_available':cuda,'missing_dependencies':missing,
        'ffmpeg_available':bool(shutil.which('ffmpeg')),'status':'ready_for_cuda' if cuda and not missing else 'cloud_gpu_required',
        'paid_resource_started_by_this_script':False,'models_downloaded_by_preflight':False}
    dump(out/'preflight.json',preflight);print(json.dumps(preflight,indent=2),flush=True)
    if not args.repair and not args.generate:return
    if not cuda:raise RuntimeError('CUDA required; no local CPU inference attempted')
    if args.repair:
        repo=args.propainter.resolve();assert (repo/'inference_propainter.py').exists()
        command=[sys.executable,str(repo/'inference_propainter.py'),'--video',str(ROOT/'context_frames'),
            '--mask',str(ROOT/'context_masks'),'--output',str(out/'repair'),'--width','256','--height','256',
            '--neighbor_length','10','--ref_stride','5','--subvideo_length','80','--mask_dilation','4','--save_frames','--fp16']
        revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
        dump(out/'repair_run.json',{'status':'started','official_repository':'https://github.com/sczhou/ProPainter','revision':revision,'command':command})
        started=time.perf_counter()
        with (out/'repair.log').open('w') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        build_conditions(ROOT,out/'repair/context_frames/frames')
        dump(out/'repair_run.json',{'status':'completed_visual_review_pending','revision':revision,'elapsed_s':time.perf_counter()-started})
        print('Repair and conditions exported. Inspect repair_audit.png and condition_audit.png before --generate.',flush=True)
        if not args.generate:return
        raise RuntimeError('Separate repair review from generation; rerun --generate after visual review')
    if missing:raise RuntimeError('Install missing dependencies in cloud environment')
    assert (out/'conditions_manifest.json').exists(), 'Run and review repaired conditions first'
    if not args.reviewed_repair:raise RuntimeError('Use --reviewed-repair only after actual visual inspection')
    if importlib.util.find_spec('hf_transfer') is None:os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
    import torch,diffusers
    from PIL import Image
    from diffusers import AnimateDiffSparseControlNetPipeline,MotionAdapter,SparseControlNetModel,AutoencoderKL,DPMSolverMultistepScheduler
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'requirements_resolved.txt').open('w'),check=True)
    models,revisions=job['models'],job['revisions'];common={'torch_dtype':torch.float16}
    motion=MotionAdapter.from_pretrained(models['motion'],revision=revisions['motion'],**common)
    control=SparseControlNetModel.from_pretrained(models['control'],revision=revisions['control'],**common)
    vae=AutoencoderKL.from_pretrained(models['vae'],revision=revisions['vae'],**common)
    pipe=AnimateDiffSparseControlNetPipeline.from_pretrained(models['base'],revision=revisions['base'],motion_adapter=motion,controlnet=control,vae=vae,**common)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,beta_schedule='linear',algorithm_type='dpmsolver++',use_karras_sigmas=True)
    pipe.load_lora_weights(models['lora'],revision=revisions['lora'],adapter_name='motion_lora')
    pipe.vae.enable_slicing();pipe.enable_model_cpu_offload()
    dump(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'diffusers':diffusers.__version__,
        'models':models,'revisions':revisions,'gpu':torch.cuda.get_device_name(0),'job':job})
    for name in job['primary_cases']:
        target=out/name;assert not target.exists(),'Use a fresh results directory, never overwrite a prior neural run'
        target.mkdir();conditions=[Image.open(out/'conditions'/name/f'{i:03d}.png').convert('RGB') for i in range(16)]
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            frames=pipe(prompt=job['prompt'],negative_prompt=job['negative_prompt'],height=384,width=384,num_frames=16,
                num_inference_steps=20,guidance_scale=6.,conditioning_frames=conditions,controlnet_frame_indices=list(range(16)),
                controlnet_conditioning_scale=1.,generator=torch.Generator('cpu').manual_seed(101)).frames[0]
            assert len(frames)==16
            export_outputs(ROOT,name,frames)
            dump(target/'run.json',{'status':'generated_visual_ball_checks_pending','elapsed_s':time.perf_counter()-started,
                'peak_vram_GiB':torch.cuda.max_memory_allocated()/2**30,'seed':101,'steps':20,
                'raw_frames_preserved':True,'no_program_ball_added_to_neural_output':True,'first_frame_exact_preservation':False})
        except Exception as error:
            dump(target/'run.json',{'status':'failed','error':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair',action='store_true');parser.add_argument('--generate',action='store_true')
    parser.add_argument('--reviewed-repair',action='store_true');parser.add_argument('--propainter',type=Path,default=Path('/workspace/ProPainter'))
    main(parser.parse_args())
