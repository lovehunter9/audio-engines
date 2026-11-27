"""Install only role-version differences while retaining the exact sealed closure."""
import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys


def normalized(version):
    return version.split('+', 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    profile = json.load(open(args.lock))['profiles'][args.role]
    code = 'import importlib.metadata as m,json\nr={}\nfor d in m.distributions():\n r.setdefault(d.metadata["Name"].lower().replace("_","-"),d.version)\nprint(json.dumps(r))'
    actual = json.loads(subprocess.check_output([args.python, '-c', code], text=True))
    required = profile['resolved_transitive_versions']
    changes = [name + '==' + version for name, version in required.items()
               if normalized(actual.get(name.lower().replace('_', '-'), 'missing')) != version]
    if changes:
        subprocess.run([args.python, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', *changes], check=True)
    actual = json.loads(subprocess.check_output([args.python, '-c', code], text=True))
    differences = {name: {'expected': version, 'actual': actual.get(name.lower().replace('_', '-'))}
                   for name, version in required.items()
                   if normalized(actual.get(name.lower().replace('_', '-'), 'missing')) != version}
    if differences:
        raise RuntimeError(json.dumps(differences))
    python = subprocess.check_output([args.python, '-c', 'import platform;print(platform.python_version())'], text=True).strip()
    if python != profile['python']:
        raise RuntimeError('Python interpreter differs from the sealed version')
    print(json.dumps({'role': args.role, 'matched': len(required), 'changed_distributions': len(changes),
                      'versions': 'Exact sealed public versions; CUDA wheel local suffixes verified separately'}))


if __name__ == '__main__':
    main()
