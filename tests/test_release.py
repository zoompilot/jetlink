import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


VERIFIER = Path(__file__).resolve().parents[1] / 'scripts/verify_release.py'
spec = importlib.util.spec_from_file_location('verify_release', VERIFIER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def package(tmp_path):
  for name in ('jetlink/client.py', 'scripts/setup.sh', 'pyproject.toml', 'docker/Dockerfile'):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
  return tmp_path


def test_release_accepts_exact_source_and_ignores_logs(package):
  digest = module.source_digest(package)
  (package / 'debug.log').write_text('runtime data')
  lock = package / 'release.json'
  lock.write_text(json.dumps({'source_sha256': digest}))
  result = subprocess.run([sys.executable, str(VERIFIER), str(package), str(lock)], capture_output=True)
  assert result.returncode == 0
  assert result.stdout.decode().strip() == digest


@pytest.mark.parametrize('change', ['modify', 'add', 'delete'])
def test_release_rejects_runtime_source_changes(package, change):
  digest = module.source_digest(package)
  lock = package / 'release.json'
  lock.write_text(json.dumps({'source_sha256': digest}))
  if change == 'modify':
    (package / 'jetlink/client.py').write_text('changed')
  elif change == 'add':
    (package / 'jetlink/extra.py').write_text('extra code')
  else:
    (package / 'jetlink/client.py').unlink()
  result = subprocess.run([sys.executable, str(VERIFIER), str(package), str(lock)], capture_output=True)
  assert result.returncode != 0
  assert b'Jetlink source mismatch' in result.stderr
