from pathlib import Path
import subprocess
import psutil


root = Path('/home/rocm-user/jupyter/nexus')
pid_path = root / 'outputs' / 'logs' / 'remote_v8_pipeline.pid'
info = {}
if pid_path.exists():
    pid = int(pid_path.read_text(encoding='utf-8').strip())
    info['pid'] = pid
    info['running'] = psutil.pid_exists(pid)
    if info['running']:
        proc = psutil.Process(pid)
        info['status'] = proc.status()
        info['cpu_percent'] = proc.cpu_percent(interval=1.0)
        info['memory_mb'] = round(proc.memory_info().rss / (1024 * 1024), 2)
        info['cmdline'] = proc.cmdline()
        try:
            children = proc.children(recursive=True)
            info['children'] = [
                {
                    'pid': child.pid,
                    'status': child.status(),
                    'cpu_percent': child.cpu_percent(interval=0.1),
                    'memory_mb': round(child.memory_info().rss / (1024 * 1024), 2),
                    'cmdline': child.cmdline(),
                }
                for child in children
            ]
        except Exception as exc:
            info['children_error'] = repr(exc)
try:
    rocm = subprocess.run(
        ['bash', '-lc', 'rocm-smi --showuse --showmemuse --showpower --showclocks'],
        capture_output=True,
        text=True,
        check=False,
    )
    info['rocm_smi'] = rocm.stdout.splitlines()[-20:]
except Exception as exc:
    info['rocm_smi_error'] = repr(exc)
print(info)
