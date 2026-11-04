"""Package source and runnable saved artifacts; validate ZIP and SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
import zipfile


def package():
    roots = ['src', 'scripts', 'tests', 'configs', 'docs', 'reports', 'data/processed',
             'artifacts/recommender/poc01', 'artifacts/baselines', 'artifacts/failed_collapse']
    files = set()
    for name in roots:
        root = Path(name)
        if not root.exists():
            raise FileNotFoundError(root)
        files.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in {'.pyc', '.tmp'})
    for name in ['README.md', 'HANDOFF.md', 'todo.md', 'pyproject.toml', 'uv.lock', '.python-version', '.gitignore']:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        files.add(path)
    files.update(p for p in Path('artifacts').iterdir() if p.is_file()
                 and p.suffix in {'.json', '.yaml', '.npy', '.pt', '.log'})
    expected = ['artifacts/recommender/poc01/best/model.safetensors',
                'artifacts/recommender/poc01/latest.pt', 'artifacts/rqvae.pt',
                'artifacts/item_embeddings.npy', 'docs/oral_exam.pptx']
    for name in expected:
        if Path(name) not in files:
            raise FileNotFoundError(name)
    output = Path('submission')
    output.mkdir(exist_ok=True)
    destination = output / 'tiger_submission.zip'
    temporary = output / 'tiger_submission.tmp.zip'
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.as_posix())
        archive.writestr('MANIFEST.sha256', ''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for name, expected_hash in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected_hash, name
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (output / 'tiger_submission.zip.sha256').write_text(f'{digest}  {destination.name}\n')
    report = {'archive': str(destination), 'files': len(hashes), 'bytes': destination.stat().st_size,
              'sha256': digest, 'crc_and_content_hashes_verified': True,
              'uploaded': False, 'upload_policy': 'Skipped at user request; user will submit.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    package()
