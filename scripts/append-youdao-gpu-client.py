"""Append only a digest-pinned official library to one architecture's wrapper layer."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request

REPOSITORY = 'beclab/vgpu_builder'
INDEX = 'sha256:ed2d0911d56dd663ee524e03a797f0cc18a3b4a59cc0d0aa9db16f6fe83f96ae'
REVISION = '934a1729c075db1c0036e8f76acd4c4f23bab43b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=['linux/amd64', 'linux/arm64'], required=True)
    parser.add_argument('--layer', type=Path, required=True)
    args = parser.parse_args()
    auth = 'https://auth.docker.io/token?' + urllib.parse.urlencode({
        'service': 'registry.docker.io', 'scope': 'repository:' + REPOSITORY + ':pull'})
    with urllib.request.urlopen(auth, timeout=30) as response:
        token = json.load(response)['token']

    def fetch(kind, digest):
        request = urllib.request.Request('https://registry-1.docker.io/v2/' + REPOSITORY + '/' + kind + '/' + digest,
            headers={'Authorization': 'Bearer ' + token, 'Accept': ','.join([
                'application/vnd.oci.image.index.v1+json', 'application/vnd.oci.image.manifest.v1+json',
                'application/vnd.docker.distribution.manifest.list.v2+json',
                'application/vnd.docker.distribution.manifest.v2+json'])})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(2 * 1024 ** 2 + 1)
        if len(raw) > 2 * 1024 ** 2 or 'sha256:' + hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError('Official artifact exceeded bound or differs from pinned digest')
        return raw

    index = json.loads(fetch('manifests', INDEX))
    arch = args.platform.split('/')[1]
    descriptor = next(r for r in index['manifests'] if r['platform']['os'] == 'linux' and r['platform']['architecture'] == arch)
    manifest = json.loads(fetch('manifests', descriptor['digest']))
    config = json.loads(fetch('blobs', manifest['config']['digest']))
    if config['architecture'] != arch or config['config']['Labels']['org.opencontainers.image.revision'] != REVISION:
        raise RuntimeError('Official library architecture/source revision differs')
    if len(manifest['layers']) != 1 or manifest['layers'][0]['size'] > 1024 ** 2:
        raise RuntimeError('Require the tiny official scratch artifact')
    layer = manifest['layers'][0]
    raw = fetch('blobs', layer['digest'])
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        member = next(r for r in archive if r.name.lstrip('./') == 'vgpu/libvgpu.so')
        data = archive.extractfile(member).read()
    machine = 62 if arch == 'amd64' else 183
    if data[:6] != b'\x7fELF\x02\x01' or int.from_bytes(data[18:20], 'little') != machine:
        raise RuntimeError('Official binary ELF architecture differs')
    if platform.machine() not in ({'x86_64', 'AMD64'} if arch == 'amd64' else {'aarch64', 'arm64'}):
        raise RuntimeError('Build the tiny loader adapter on its native CI runner')
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / 'client-audit.so'
        subprocess.run(['cc', '-shared', '-fPIC', '-O2', '-Werror', '-Wall', '-o', str(target),
                        str(Path(__file__).with_name('youdao-gpu-audit.c'))], check=True)
        audit = target.read_bytes()
    if int.from_bytes(audit[18:20], 'little') != machine:
        raise RuntimeError('Child-loader adapter architecture differs')
    identity = {'image': REPOSITORY + '@' + INDEX, 'platform': args.platform,
                'revision': REVISION, 'layer_digest': layer['digest'],
                'library_SHA256': hashlib.sha256(data).hexdigest(), 'library_bytes': len(data),
                'audit_SHA256': hashlib.sha256(audit).hexdigest()}
    prefix = 'app/wrapper/s2st/gpu-client/'
    with tarfile.open(args.layer, 'a') as archive:
        for name, content in [('libvgpu.so', data), ('client-audit.so', audit),
                              ('client.json', (json.dumps(identity, indent=2) + '\n').encode())]:
            item = tarfile.TarInfo(prefix + name)
            item.size, item.mode, item.uid, item.gid, item.mtime = len(content), 0o644, 0, 0, 0
            archive.addfile(item, io.BytesIO(content))
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
