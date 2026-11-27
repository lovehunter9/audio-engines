"""Fail closed on stale packaged sources before any model worker starts."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    for relative, expected in manifest['files_sha256'].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Packaged source differs from the measured export: ' + relative)
    return {'state': 'packaged_sources_match_export', 'family': manifest['family'],
            'profile_sha256': manifest['profile_sha256'], 'source_files': len(manifest['files_sha256'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory)))
