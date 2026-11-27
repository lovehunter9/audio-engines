"""Verify runtime source archives against pinned clean official Git checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/olares/s2st-original-check-20261123')
REPOSITORIES = {
    'Hy-SimulMT': ('7efcb18c521a7025b37d4bc2230a1df8ade83950', ['SimulInfer/simulinfer']),
    'Confucius4-T3PO': ('0ac9d0c806f8f139ebaa83d394ffbd5057ff3ae3', ['inference']),
    'Confucius4-R2T2': ('62757213befa282e7fbfe1bfde3c29577781527c', ['r2t2', 'ws_server.py']),
}


def paths(source, prefixes):
    files = set()
    for prefix in prefixes:
        path = source / prefix
        entries = [path] if path.is_file() else path.rglob('*')
        for file in entries:
            if file.is_file() and file.suffix in ['.py', '.toml', '.js', '.html', '.css']:
                files.add(str(file.relative_to(source)))
    return files


def manifest():
    result = {'scope': 'Runtime source only; no weights or dependency binaries', 'repositories': {}}
    for repo, (revision, prefixes) in REPOSITORIES.items():
        source = ROOT / 'sources' / repo
        head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if head != revision:
            raise RuntimeError('Official checkout revision differs: ' + repo)
        subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--', *prefixes], check=True)
        tracked = set(subprocess.check_output(['git', '-C', str(source), 'ls-files', '--', *prefixes], text=True).splitlines())
        selected = paths(source, prefixes)
        if selected - tracked:
            raise RuntimeError('Untracked runtime source present: ' + repo)
        files = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in sorted(selected)}
        result['repositories'][repo] = {'revision': revision, 'prefixes': prefixes, 'files': files,
                                        'reference_git_clean': True}
    return result


def verify(lock):
    expected = json.loads(Path(lock).read_text())
    for repo, item in expected['repositories'].items():
        source = ROOT / 'sources' / repo
        if paths(source, item['prefixes']) != set(item['files']):
            raise RuntimeError('Runtime source file set differs: ' + repo)
        for name, digest in item['files'].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Runtime source content differs: ' + repo + '/' + name)
    return {'state': 'locked_official_runtime_source_archives_match',
            'repositories': {repo: {'revision': item['revision'], 'files': len(item['files'])}
                             for repo, item in expected['repositories'].items()},
            'Git_metadata_required_on_target': False, 'model_inference_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['manifest', 'verify'], required=True)
    parser.add_argument('--lock')
    args = parser.parse_args()
    print(json.dumps(manifest() if args.mode == 'manifest' else verify(args.lock)))
