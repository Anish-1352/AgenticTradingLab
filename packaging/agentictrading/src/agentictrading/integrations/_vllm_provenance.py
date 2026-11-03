"""Credential-free source identity and artifact checksums for offline research."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess


def capture_source_provenance(repo_root):
    root = Path(repo_root)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
    try:
        commit = git('rev-parse', 'HEAD').decode().strip()
        status = git('status', '--porcelain=v1', '--untracked-files=all').decode()
        patch = git('diff', 'HEAD', '--binary')
        # Hash source bytes without persisting untracked contents or credentials.
        untracked = {}
        for name in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0'):
            p = root / name
            if name and p.is_file() and p.suffix in {'.py', '.sh', '.rst', '.md', '.toml', '.json', '.yaml', '.yml'}:
                untracked[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {'atl_source_commit': commit, 'dirty_working_tree': bool(status),
                'git_status_porcelain': status.splitlines(),
                'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
                'untracked_source_sha256': untracked,
                'source_snapshot': 'PATCH_HASH_AND_UNTRACKED_SOURCE_HASHES'}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'atl_source_commit': 'UNKNOWN', 'source_snapshot': 'NOT_CAPTURED',
                'source_provenance': 'INCOMPLETE', 'capture_error_type': type(exc).__name__}


def write_artifact_checksums(directory):
    root = Path(directory)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifact-checksums.json'}
    (root / 'artifact-checksums.json').write_text(json.dumps({'algorithm': 'SHA-256', 'files': files},
                                                          indent=2, sort_keys=True) + '\n')
