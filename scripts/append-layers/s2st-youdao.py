import argparse
import tarfile
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--layer', required=True)
parser.add_argument('--platform', required=True)
args = parser.parse_args()
family = Path(__file__).stem.removeprefix('s2st-')
with tarfile.open(args.layer, 'a') as archive:
    directories = ['home/olares', 'home/olares/s2st-original-check-20261123']
    if family == 'youdao':
        base = 'app/wrapper/s2st/locked_payload/youdao/youdao'
        directories += [base + '/wav_tmp_store', base + '/logs', base + '/logs/requests']
    for name in directories:
        item = tarfile.TarInfo(name)
        item.type, item.mode, item.uid, item.gid = tarfile.DIRTYPE, 0o755, 1000, 1000
        archive.addfile(item)
    if family == 'tencent':
        base = 'app/wrapper/s2st/locked_payload/tencent/official/Hy-SimulMT/SimulInfer/simulinfer/models'
        item = tarfile.TarInfo(base)
        item.type, item.mode, item.uid, item.gid = tarfile.DIRTYPE, 0o755, 1001, 1001
        archive.addfile(item)
        for name, folder in [('firered_stream_vad', 'Stream-VAD'), ('firered_vad', 'VAD')]:
            item = tarfile.TarInfo(base + '/' + name)
            item.type, item.mode, item.uid, item.gid = tarfile.SYMTYPE, 0o777, 1001, 1001
            item.linkname = '/home/olares/s2st-original-check-20261123/models/Tencent-VAD/' + folder
            archive.addfile(item)
if family == 'youdao':
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'append-youdao-gpu-client.py'),
                    '--platform', args.platform, '--layer', args.layer], check=True)
