"""K1–K4/C0: read-only image inspection; run with the image Python, no installs or candidate execution."""

def main():
    import concurrent.futures, datetime, importlib.metadata, json, os, pathlib, platform, shutil, subprocess, sys
    def run(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=25)
            return dict(argv=args, returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
        except Exception as e:
            return dict(argv=args, error=type(e).__name__ + ': ' + str(e))
    def read(path):
        try: return pathlib.Path(path).read_text().strip()
        except OSError as e: return str(e)
    commands = {
     'cpu': ['lscpu'],
     'ram': ['free', '-b'],
     'disk': ['df', '-h', '/', '/root/autodl-tmp'],
     'nvidia_smi': ['nvidia-smi'],
     'nvcc': ['nvcc', '--version'],
     'compiler': ['g++', '--version'],
     'ninja': ['ninja', '--version'],
     'torch': [sys.executable, '-c', 'import json,torch; print(json.dumps(dict(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),cudnn=torch.backends.cudnn.version(),compiled_arches=torch.cuda.get_arch_list())))'],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
     results=dict(zip(commands,pool.map(run,commands.values())))
    packages={}
    for name in ['torch','triton','vllm','transformers','trl','peft','safetensors','modelscope','numpy']:
     try: packages[name]=importlib.metadata.version(name)
     except importlib.metadata.PackageNotFoundError: packages[name]=None
    cuda_paths={}
    for p in pathlib.Path('/usr/local').glob('cuda*'):
     cuda_paths[str(p)]={'target':str(p.resolve()),'version_json':read(p/'version.json') if (p/'version.json').exists() else None,'nvcc_exists':(p/'bin/nvcc').is_file(),'headers_exist':(p/'include/cuda_runtime.h').is_file()}
    r={
     'scope':'remote_readonly_inventory_not_C0',
     'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'os_release':read('/etc/os-release'),'kernel':platform.release(),
     'python':sys.version,'python_executable':sys.executable,'logical_cpus_visible':os.cpu_count(),
     'cpu_affinity':len(os.sched_getaffinity(0)),
     'cgroup_limits':{p:read('/sys/fs/cgroup/'+p) for p in ['cpu.max','memory.max','pids.max']},
     'packages':packages,'commands':results,'cuda_paths':cuda_paths,
     'tools':{n:shutil.which(n) for n in ['python','python3','pip','uv','nvcc','gcc','g++','ninja','git','curl','bwrap','docker']},
     'nvidia_devices':[str(p) for p in pathlib.Path('/dev').glob('nvidia*')],
     'host_driver':read('/proc/driver/nvidia/version'),
     'gpu_env':{n:os.environ.get(n) for n in ['CUDA_HOME','CUDA_PATH','CONDA_PREFIX','NVIDIA_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']},
     'project_dir_exists':pathlib.Path('/root/autodl-tmp/cuda-gemma4-llm').exists()
    }
    print('BEGIN_INVENTORY_JSON')
    print(json.dumps(r,indent=2))
    print('END_INVENTORY_JSON')

if __name__ == "__main__":
    main()
