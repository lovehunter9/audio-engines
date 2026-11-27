"""Filesystem-only image aliases; no runtime model or synthesis policy changes."""
import argparse
from pathlib import Path
import subprocess
import sysconfig


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError('Unexpected existing image-layout path: ' + str(path))
    path.symlink_to(target)


parser = argparse.ArgumentParser()
parser.add_argument('--family', choices=['tencent', 'youdao'], required=True)
family = parser.parse_args().family
site = Path(sysconfig.get_paths()['purelib'])
frontend = Path('/opt/venvs/frontend/lib/python3.12/site-packages')
link(Path('/opt/audio-locked'), '/app/wrapper/s2st/locked_payload/' + family)
link(Path('/opt/venvs/ming-overlay'), site)
link(Path('/opt/cuda130-toolchain/nvidia/cu13'), site / 'nvidia/cu13')
link(Path('/opt/compat-python-headers/include/python3.12'), '/usr/include/python3.12')
link(Path('/opt/compat-python-headers/include/x86_64-linux-gnu/python3.12'),
     Path('/usr/include') / subprocess.check_output(['gcc', '-print-multiarch'], text=True).strip() / 'python3.12')
(frontend / 'locked-s2st.pth').write_text('/opt/audio-locked/common\n/opt/audio-locked/official/Hy-SimulMT/SimulInfer\n')
feature = frontend / 'simulst_mm/model'
feature.mkdir(parents=True)
link(feature / 'processing.py', '/opt/audio-locked/official/simulst/simulst_mm/model/processing.py')
client = Path('/opt/venvs/frontend/bin/simulinfer-client')
client.write_text('#!/opt/venvs/frontend/bin/python\nfrom simulinfer.client import main\nmain()\n')
client.chmod(0o755)
libraries = sorted({str(path.parent) for path in site.glob('nvidia/**/*.so*') if path.is_file()})
if not (site / 'nvidia/cu13/include/cuda.h').is_file():
    raise RuntimeError('Frozen CUDA13 compiler headers are absent')
Path('/etc/ld.so.conf.d/locked-s2st.conf').write_text('/opt/tencent-native\n/opt/youdao-native\n' + '\n'.join(libraries) + '\n')
subprocess.run(['ldconfig'], check=True)
