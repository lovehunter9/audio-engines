"""Select the official handoff client for Youdao ASR/MT child processes only."""
import hashlib
import json
from pathlib import Path
import re


CLIENT = Path('/app/wrapper/s2st/gpu-client/libvgpu.so')


def frontend_environment(environment, client=CLIENT, preload_file=Path('/etc/ld.so.preload')):
    """Preserve the platform allocation and every unrelated preload library."""
    result = dict(environment)
    preload = re.split(r'[\s:]+', result.get('LD_PRELOAD', '').strip())
    platform = [p for p in preload if Path(p).name == 'libvgpu.so']
    if preload_file.is_file():
        platform += [p for p in preload_file.read_text().split() if Path(p).name == 'libvgpu.so']
    if not platform:
        return result
    data = client.read_bytes()
    identity = json.loads(client.with_name('client.json').read_text())
    if hashlib.sha256(data).hexdigest() != identity['library_SHA256']:
        raise RuntimeError('Packaged Youdao GPU client identity differs')
    auditor = client.with_name('client-audit.so')
    if hashlib.sha256(auditor.read_bytes()).hexdigest() != identity['audit_SHA256']:
        raise RuntimeError('Packaged Youdao child-loader adapter identity differs')
    result['LD_PRELOAD'] = ':'.join([p for p in preload if p and p not in platform])
    if not preload_file.is_file():
        result['LD_PRELOAD'] = ':'.join([str(client), result['LD_PRELOAD']]).rstrip(':')
    result['LD_AUDIT'] = ':'.join([str(auditor), *filter(None, result.get('LD_AUDIT', '').split(':'))])
    result.update(OLARES_TS_IDLE_MS='50', OLARES_TS_IDLE_RELEASE_MS='50',
                  OLARES_TS_SYNC_ON_RELEASE='1', OLARES_TS_PREFETCH='0')
    return result
