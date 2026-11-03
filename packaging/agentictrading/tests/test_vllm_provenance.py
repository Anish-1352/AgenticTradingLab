"""Source/configuration provenance for future external runs; no GPU needed."""
import hashlib
import json
from pathlib import Path
import subprocess
from agentictrading.integrations._vllm_provenance import capture_source_provenance, write_artifact_checksums


def test_source_identity_captures_commit_dirty_patch_and_untracked_code(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','config','user.email','test@example.invalid'],cwd=tmp_path,check=True)
    subprocess.run(['git','config','user.name','test'],cwd=tmp_path,check=True)
    code=tmp_path/'client.py';code.write_text('x = 1\n')
    subprocess.run(['git','add','client.py'],cwd=tmp_path,check=True)
    subprocess.run(['git','commit','-qm','baseline'],cwd=tmp_path,check=True)
    clean=capture_source_provenance(tmp_path)
    assert len(clean['atl_source_commit'])==40 and clean['dirty_working_tree'] is False
    code.write_text('x = 2\n')
    private=tmp_path/'untracked.py';private.write_text('not_an_actual_key = "private-test-value"\n')
    dirty=capture_source_provenance(tmp_path)
    assert dirty['atl_source_commit']==clean['atl_source_commit']
    assert dirty['dirty_working_tree'] is True
    assert dirty['source_patch_sha256']!=clean['source_patch_sha256']
    assert dirty['untracked_source_sha256']['untracked.py']==hashlib.sha256(private.read_bytes()).hexdigest()
    assert 'private-test-value' not in json.dumps(dirty)
    assert dirty['source_snapshot']=='PATCH_HASH_AND_UNTRACKED_SOURCE_HASHES'


def test_artifact_checksums_match_original_bytes_without_self_hash(tmp_path):
    p=tmp_path/'inference-smoke.json';p.write_bytes(b'{"verified":true}\n')
    log=tmp_path/'vllm-server.log';log.write_bytes(b'original bytes\r\n')
    write_artifact_checksums(tmp_path)
    first=(tmp_path/'artifact-checksums.json').read_bytes()
    data=json.loads(first)
    assert data['files']['vllm-server.log']==hashlib.sha256(log.read_bytes()).hexdigest()
    assert 'artifact-checksums.json' not in data['files']
    write_artifact_checksums(tmp_path)
    assert (tmp_path/'artifact-checksums.json').read_bytes()==first
