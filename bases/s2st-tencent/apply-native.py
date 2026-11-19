import hashlib
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
source = root / 'tools/mtmd/mtmd-cli.cpp'
expected = '885e901df3703a565040b13a3eb3a7c816c4649711b77f88a86351f448433ed6'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Pinned official Tencent CLI source changed')
subprocess.run(['patch', '-p1', '--batch', '--forward', '-i', str(Path(__file__).with_name('native-io.patch').resolve())], cwd=root, check=True)
