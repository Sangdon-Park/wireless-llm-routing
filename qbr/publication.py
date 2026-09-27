"""Check source files against archived or published snapshot checksums."""
import hashlib
import json


def verify_source(root, name, archived_digest):
    content = (root / name).read_bytes().replace(b'\r\n', b'\n')
    actual = hashlib.sha256(content).hexdigest()
    if actual == archived_digest:
        return
    manifest = json.loads((root / 'confirmation/publication_manifest.json').read_text(encoding='utf-8'))
    record = manifest['source_files_lf'][name]
    assert record['archived_sha256'] == archived_digest, name
    assert record['published_sha256'] == actual, name
