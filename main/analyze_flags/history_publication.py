"""Stage history artifacts and preserve the last complete files on failure."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _atomic_write(path, write):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_write_csv(frame, path):
    _atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False))


def atomic_write_json(value, path):
    def write(temporary):
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    _atomic_write(path, write)


def file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_publication_complete(artifact_dir):
    journal = Path(artifact_dir) / '.history-publication.json'
    if journal.exists():
        raise RuntimeError(
            f'Interrupted history publication requires recovery: {journal}. '
            'Refusing to read artifacts that may belong to different runs.')


def publish_staged_files(artifact_dir, stage_dir, names, expected_manual):
    """Publish fully serialized files; roll back replacements on any error.

    A durable journal and sibling backups survive a process interruption.
    An interrupted publication blocks another run until its saved files have
    been recovered, rather than silently treating a mixed set as last-good.
    """
    root = Path(artifact_dir).resolve()
    stage = Path(stage_dir).resolve()
    names = list(dict.fromkeys(names))
    if any(Path(name).name != name for name in names):
        raise ValueError('History publication requires artifact basenames')
    for name in names:
        if not (stage / name).is_file():
            raise RuntimeError(f'Missing staged history artifact: {name}')
    for name, expected in expected_manual.items():
        if file_fingerprint(root / name) != expected:
            raise RuntimeError(
                f'{name} changed during collection; rerun to use the new '
                'review decisions or mappings. No artifacts were published.')

    journal = root / '.history-publication.json'
    if journal.exists():
        raise RuntimeError(
            f'Interrupted history publication requires recovery: {journal}')
    backup = Path(tempfile.mkdtemp(prefix='.history-backup-', dir=root))
    existed = {}
    replaced = []
    completed = False
    try:
        for name in names:
            existed[name] = (root / name).exists()
            if existed[name]:
                shutil.copy2(root / name, backup / name)
        atomic_write_json({'backup_directory': str(backup),
                           'artifacts': existed}, journal)
        try:
            for name, expected in expected_manual.items():
                if file_fingerprint(root / name) != expected:
                    raise RuntimeError(
                        f'{name} changed during collection (while preparing publication); '
                        'rerun with the current review decisions or mappings.')
            for name in names:
                os.replace(stage / name, root / name)
                replaced.append(name)
        except BaseException as failure:
            rollback_errors = []
            for name in reversed(replaced):
                try:
                    if existed[name]:
                        os.replace(backup / name, root / name)
                    else:
                        (root / name).unlink()
                except OSError as error:
                    rollback_errors.append(f'{name}: {error}')
            if rollback_errors:
                raise RuntimeError(
                    f'History publication failed; recovery files remain at '
                    f'{backup}. Rollback failed: {"; ".join(rollback_errors)}'
                ) from failure
            completed = True
            raise
        completed = True
    finally:
        if completed:
            journal.unlink(missing_ok=True)
        # Verify the generated backup directory before recursively removing it.
        if (completed or not journal.exists()) and backup.parent == root \
                and backup.name.startswith('.history-backup-'):
            shutil.rmtree(backup)
