from pathlib import Path
import subprocess


root = Path('/home/rocm-user/jupyter/nexus')
log_dir = root / 'outputs' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / 'remote_v8_pipeline.log'
pid_path = log_dir / 'remote_v8_pipeline.pid'
log_handle = log_path.open('ab')
proc = subprocess.Popen(
    ['python', 'scripts/remote_v8_train.py'],
    cwd=root,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
pid_path.write_text(str(proc.pid), encoding='utf-8')
print({'pid': proc.pid, 'log_path': str(log_path), 'pid_path': str(pid_path)})
