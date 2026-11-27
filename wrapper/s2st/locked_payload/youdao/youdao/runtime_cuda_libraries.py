"""Resolve actual locked CUDA wheel libraries in isolated/readonly environments."""
import importlib.metadata as metadata
from pathlib import Path


def library_directories():
    directories = set()
    found_runtime = False
    for distribution in metadata.distributions():
        name = distribution.metadata['Name'].lower().replace('_', '-')
        if not name.startswith('nvidia-'):
            continue
        for item in distribution.files or []:
            if '.so' not in Path(str(item)).name:
                continue
            path = Path(distribution.locate_file(item))
            if path.is_file():
                directories.add(str(path.parent))
                if path.name.startswith('libcudart.so.13'):
                    found_runtime = True
    if not found_runtime:
        raise RuntimeError('The locked CUDA13 runtime library is not available')
    return sorted(directories)
