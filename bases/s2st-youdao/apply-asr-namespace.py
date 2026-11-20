"""Isolate ASR dependencies without changing canonical MING packages."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def change(text, before, after, expected=1):
    if text.count(before) != expected:
        raise RuntimeError('Pinned optional import changed: ' + before)
    return text.replace(before, after)


def optional_imports(name, text):
    if name == '__init__.py':
        text = change(text, 'from .inference.qwen3_forced_aligner import Qwen3ForcedAligner\n', '')
        return text + '\n\ndef __getattr__(name):\n    if name == "Qwen3ForcedAligner":\n        from .inference.qwen3_forced_aligner import Qwen3ForcedAligner\n        return Qwen3ForcedAligner\n    raise AttributeError(name)\n'
    if name == 'inference/qwen3_asr.py':
        text = change(text, 'from .qwen3_forced_aligner import Qwen3ForcedAligner\n', '')
        text = change(text, 'forced_aligner: Optional[Qwen3ForcedAligner] = None,', 'forced_aligner: Optional[Any] = None,')
        return change(text, '        if forced_aligner is not None:\n', '        if forced_aligner is not None:\n            from .qwen3_forced_aligner import Qwen3ForcedAligner\n', 2)
    if name == 'inference/utils.py':
        text = change(text, 'import librosa\n', '')
        text = change(text, '        audio, sr = librosa.load(x, sr=None, mono=False)', '        import librosa\n        audio, sr = librosa.load(x, sr=None, mono=False)')
        return change(text, '    if sr != SAMPLE_RATE:\n', '    if sr != SAMPLE_RATE:\n        import librosa\n')
    return text


def main(args):
    args.output.mkdir(exist_ok=False)
    aliases = {'transformers': 'asr_transformers', 'tokenizers': 'asr_tokenizers'}
    audit = []
    for package, target, version in [('transformers', 'asr_transformers', '4.57.6'),
                                     ('tokenizers', 'asr_tokenizers', '0.22.2'),
                                     ('qwen_asr', 'qwen_asr', '0.0.6')]:
        shutil.copytree(args.source/package, args.output/target,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for file in (args.output/target).rglob('*.py'):
            original = file.read_bytes().decode('utf-8')
            optional = optional_imports(str(file.relative_to(args.output/target)), original) if package == 'qwen_asr' else original
            rewritten = optional
            if package != 'tokenizers':
                for before, after in aliases.items():
                    rewritten = re.sub(r'\b'+before+r'\b', after, rewritten)
            inverse = rewritten
            if package != 'tokenizers':
                for before, after in aliases.items():
                    inverse = re.sub(r'\b'+after+r'\b', before, inverse)
            if inverse != optional:
                raise RuntimeError('Namespace transformation is not reversible: '+str(file))
            compile(rewritten, str(file), 'exec')
            file.write_bytes(rewritten.encode())
            audit.append(dict(path=str(file.relative_to(args.output)),
                              original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                              optional_sha256=hashlib.sha256(optional.encode()).hexdigest(),
                              final_sha256=hashlib.sha256(rewritten.encode()).hexdigest()))
        metadata = list(args.source.glob(package+'-'+version+'.dist-info'))
        if len(metadata) != 1:
            raise RuntimeError('Expected one pinned distribution: '+package)
        destination = args.output/(target+'-'+version+'.dist-info')
        shutil.copytree(metadata[0], destination)
        if target != package:
            p = destination/'METADATA'
            text = p.read_text()
            text = re.sub(r'(?m)^Name: .*$', 'Name: '+target.replace('_', '-'), text, count=1)
            for before, after in aliases.items():
                text = re.sub(r'(?m)^(Requires-Dist: )'+before+r'\b', r'\g<1>'+after.replace('_','-'), text)
            p.write_text(text)
        for filename in ['RECORD', 'direct_url.json', 'INSTALLER', 'REQUESTED']:
            (destination/filename).unlink(missing_ok=True)
    (args.output/'namespace-audit.json').write_text(json.dumps(dict(
        namespace_only=aliases, canonical_ming_dependencies_changed=False,
        optional_import_scope='Forced aligner and Librosa imports are lazy; 16kHz array ASR path is unchanged',
        files=audit), indent=2))
    print(json.dumps(dict(python_files=len(audit), directory=str(args.output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
