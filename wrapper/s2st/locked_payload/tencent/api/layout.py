"""Container-only path aliases preserve the measured runtime source bytes."""
import json
from pathlib import Path

ROOT = Path('/home/olares/s2st-original-check-20261123')
PACKAGE = Path('/opt/audio-locked')


def alias(destination, source):
    destination, source = Path(destination), Path(source)
    if not source.exists():
        raise RuntimeError('Required packaged/model path is missing: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError('Existing runtime alias points elsewhere: ' + str(destination))
        return
    if destination.exists():
        raise RuntimeError('Refusing to replace an existing non-alias path: ' + str(destination))
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare(family, cache, models):
    # This marker is created only by the image recipe, preventing accidental
    marker = PACKAGE / 'manifest.json'
    if not marker.is_file():
        raise RuntimeError('Locked service layout is container-only')
    manifest = json.loads(marker.read_text())
    if manifest['family'] != family:
        raise RuntimeError('Image family differs from selected runtime profile')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ['runs', 'cache']:
        target = cache / name
        target.mkdir(exist_ok=True)
        alias(ROOT / name, target)
    # The unchanged excluded warm-up reads the same160000-source-sample prefix.
    alias(ROOT / 'inputs', cache / 'warmup-inputs')
    alias(ROOT / 'scripts', PACKAGE / 'common')
    alias(ROOT / 'envs/tencent-model', '/opt/venvs/frontend')
    alias(ROOT / 'envs/ming-overlay', '/opt/venvs/ming-overlay')
    alias(ROOT / 'envs/cuda130-toolchain', '/opt/cuda130-toolchain')
    alias(ROOT / 'envs/python-build-headers', '/opt/compat-python-headers')
    for name in ['Hy-SimulMT', 'Confucius4-T3PO', 'Confucius4-R2T2']:
        alias(ROOT / 'sources' / name, PACKAGE / 'official' / name)
    alias(ROOT / 'sources/Ming-frozen', PACKAGE / 'ming-native')
    alias(ROOT / 'models/Ming', models['speech'])
    snapshot = ROOT / 'sources/quantized-003-snapshot'
    (snapshot / 'results').mkdir(parents=True, exist_ok=True)
    env_source = PACKAGE / 'native-loader-env.json'
    if not env_source.is_file():
        raise RuntimeError('Declared image native loader environment is missing')
    alias(snapshot / 'results/runtime-env.json', env_source)
    (ROOT / 'preflight').mkdir(exist_ok=True)
    alias(ROOT / 'preflight/official-runtime-source-files-20261125.json', PACKAGE / 'official-runtime-source-files-20261125.json')
    if family == 'tencent':
        alias(ROOT / 'models/Tencent-VAD', models['vad'])
        alias(ROOT / 'models/Hy-SimulMT-30B-A3B', models['translation_meta'])
        alias(snapshot / 'models/Hy-SimulMT-30B-A3B-Q4_K_M.gguf', models['translation'])
        alias(ROOT / 'results/conformant-tencent-native-20261125-r4', '/opt/tencent-native')
        alias(snapshot / 'sources/llama.cpp/build/bin', '/opt/tencent-native')
        reference = ROOT / 'runs/official-tencent-filter-active-control2/measurements'
        reference.mkdir(parents=True, exist_ok=True)
        alias(reference / 'config.toml', PACKAGE / 'tencent-reference.toml')
    elif family == 'youdao':
        alias(ROOT / 'envs/youdao-asr', '/opt/venvs/youdao-asr')
        alias(ROOT / 'models/r2t2-bf16', models['recognition'])
        alias(ROOT / 'models/t3po-bf16', models['translation_meta'])
        alias(ROOT / 'models/FireRedVAD/Stream-VAD', models['vad'])
        alias(ROOT / 'models/quantized/Confucius4-T3PO-Q5_K_M.gguf', models['translation'])
        alias(ROOT / 'results/revised-youdao-native-20261125', '/opt/youdao-native')
        alias(snapshot / 'youdao/sources/llama.cpp/build/bin', '/opt/youdao-native')
    else:
        raise ValueError('Unknown profile family')
    return manifest
