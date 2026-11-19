import hashlib
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
source = root / 'tools/mtmd/mtmd-cli.cpp'
expected = '885e901df3703a565040b13a3eb3a7c816c4649711b77f88a86351f448433ed6'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise RuntimeError('Pinned official Tencent CLI source changed')
# Apply only exact unified hunks; the runtime parent intentionally has no patch binary.
original = source.read_text().splitlines(keepends=True)
patch = Path(__file__).with_name('native-io.patch').read_text().splitlines(keepends=True)
result = []
position = 0
index = 0
while index < len(patch):
    header = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', patch[index])
    if header is None:
        index += 1
        continue
    start = int(header[1]) - 1
    if start < position:
        raise RuntimeError('Overlapping native patch hunks')
    result.extend(original[position:start])
    position = start
    removed = added = 0
    index += 1
    while index < len(patch) and not patch[index].startswith('@@ '):
        line = patch[index]
        if line[:1] not in (' ', '+', '-'):
            raise RuntimeError('Unexpected native patch content')
        if line[0] in (' ', '-'):
            if position >= len(original) or original[position] != line[1:]:
                raise RuntimeError('Native patch context differs from the pinned source')
            position += 1
            removed += 1
        if line[0] in (' ', '+'):
            result.append(line[1:])
            added += 1
        index += 1
    if removed != int(header[2] or 1) or added != int(header[4] or 1):
        raise RuntimeError('Native patch hunk count mismatch')
result.extend(original[position:])
source.write_text(''.join(result))

if len(sys.argv) > 2 and sys.argv[2] == "--shared-entry":
    patched = source.read_text()
    if patched.count("std::cin") != 3:
        raise RuntimeError("Pinned PCM transport changed")
    (root / "shared-cli.cpp").write_text(patched.replace("std::cin", "audio_s2st_input()"))
