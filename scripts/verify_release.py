#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Print a runtime source digest, or verify it against a fork's release lock.
"""
import hashlib
import json
import sys
from pathlib import Path


def source_digest(root: Path) -> str:
  files = set(root.joinpath('jetlink').rglob('*.py'))
  files.update(p for p in root.joinpath('scripts').iterdir() if p.is_file())
  files.update([root / 'pyproject.toml', root / 'docker/Dockerfile'])
  digest = hashlib.sha256()
  for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
    name = path.relative_to(root).as_posix().encode()
    data = path.read_bytes()
    digest.update(len(name).to_bytes(8, 'big') + name)
    digest.update(len(data).to_bytes(8, 'big') + data)
  return digest.hexdigest()


if __name__ == '__main__':
  actual = source_digest(Path(sys.argv[1]))
  if len(sys.argv) > 2:
    expected = json.loads(Path(sys.argv[2]).read_text())['source_sha256']
    if actual != expected:
      raise SystemExit(f'Jetlink source mismatch: expected {expected}, found {actual}')
  print(actual)
