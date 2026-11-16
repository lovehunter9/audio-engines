"""Keep runtime contents intact while bounding ordinary registry blob sizes."""
import json
import os
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
root = Path('/opt/runtime-parts')
limit, count = 1024**3, 16
parts = [root/f'{i:02d}' for i in range(count)]
for part in parts:
    part.mkdir(parents=True, exist_ok=True)
entries = []
for directory, dirs, files in os.walk(source, followlinks=False):
    base = Path(directory)
    links = [name for name in dirs if (base/name).is_symlink()]
    dirs[:] = [name for name in dirs if name not in links]
    for name in files+links:
        path = base/name
        entries.append((0 if path.is_symlink() else path.stat().st_size, path))
sizes, totals = [0]*count, [0]*count
for size, path in sorted(entries, key=lambda item: (-item[0], str(item[1]))):
    choices = [i for i in range(count) if sizes[i]+size <= limit or totals[i] == 0]
    if not choices:
        raise RuntimeError('Runtime exceeds the configured layer partition bound')
    index = min(choices, key=lambda i: sizes[i])
    destination = parts[index]/path.relative_to('/')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), destination)
    sizes[index] += size
    totals[index] += 1
report = {'source': str(source), 'regular_file_bytes': sum(sizes), 'entries': len(entries),
          'layers': [{'name': part.name, 'bytes': size, 'entries': total}
                     for part, size, total in zip(parts, sizes, totals)]}
Path('/opt/runtime-layout.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
