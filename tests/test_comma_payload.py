"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's payload: jetlink/, plus the root LICENSE, pyproject.toml and
README.md. A fork may vendor exactly those (StarPilot's sync tool will, with
an include list), so everything the comma imports or runs has to be under
jetlink/. Walked from the comma's entry points through every jetlink import
in the source, a function's lazy ones too, and then proved by importing the
entry points from a copy of the payload alone.
"""
from __future__ import annotations

import ast
import functools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'jetlink'
PAYLOAD = ('jetlink', 'LICENSE', 'pyproject.toml', 'README.md')

# What the comma starts or imports: the owner (python -m), the provisioning run
# it starts (python -m), what modeld and the fork's hook lines and adapter
# import, and the kit the fork's tests import
ENTRIES = (
  'jetlink.openpilot.owner',
  'jetlink.openpilot.provision',
  'jetlink.openpilot',
  'jetlink.openpilot.joining',
  'jetlink.openpilot.model_state',
  'jetlink.openpilot.warp',
  'jetlink.openpilot.hooks',
  'jetlink.openpilot.stock',
  'jetlink.openpilot.events',
  'jetlink.openpilot.testing',
)

# What the payload may import from outside itself: the standard library, what
# the comma's venv has, and the fork's tinygrad, which modeld brings. onnx is
# only behind spec_from_onnx, a developer tool
OUTSIDE = set(sys.stdlib_module_names) | {'numpy', 'tinygrad', 'onnx'}
# the fork's openpilot, which only the base adapter imports, inside its methods
HOST = {'jetlink.openpilot.stock': {'openpilot'}}

MODULE_NAME = re.compile(r'^jetlink(\.\w+)+$')


def module_file(name: str) -> Path | None:
  parts = name.split('.')
  base = ROOT.joinpath(*parts)
  for f in (base / '__init__.py', base.with_suffix('.py')):
    if f.is_file():
      return f
  return None


def imports_of(name: str, tree: ast.Module) -> tuple[set[str], set[str]]:
  """The jetlink modules `tree` imports (or names as a string, as the owner
  names the provisioning run), and the top-level names of everything else."""
  is_package = module_file(name).name == '__init__.py'
  ours, theirs = set(), set()

  def add(target: str):
    top = target.partition('.')[0]
    if top == 'jetlink':
      ours.add(target)
    else:
      theirs.add(top)

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        add(alias.name)
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        here = name.split('.')
        base = here[:len(here) - node.level + (1 if is_package else 0)]
        source = '.'.join(base + ([node.module] if node.module else []))
      else:
        source = node.module
      add(source)
      if source.partition('.')[0] == 'jetlink':
        for alias in node.names:
          # from jetlink.x import y: y is a module or a name in x
          if module_file(f'{source}.{alias.name}'):
            ours.add(f'{source}.{alias.name}')
    elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_NAME.match(node.value):
      if module_file(node.value):
        ours.add(node.value)
  return ours, theirs


@functools.cache
def graph() -> dict[str, tuple[Path, ast.Module, set[str]]]:
  """Every module on the comma's graph: its file, its tree, what else it imports."""
  modules = {}
  todo = list(ENTRIES)
  while todo:
    name = todo.pop()
    if name in modules:
      continue
    f = module_file(name)
    assert f is not None, f'{name} is imported on the comma and is not a module under jetlink/'
    tree = ast.parse(f.read_text(), str(f))
    ours, theirs = imports_of(name, tree)
    modules[name] = (f, tree, theirs)
    # a module's packages run first
    parents = {'.'.join(name.split('.')[:i]) for i in range(1, name.count('.') + 1)}
    todo += sorted((ours | parents) - modules.keys())
  return modules


def test_the_walk_reaches_the_comma_side():
  for name in ('jetlink.comma.owner', 'jetlink.comma.root', 'jetlink.transport.ffs', 'jetlink.client',
               'jetlink.registry.index', 'jetlink.openpilot.provision'):
    assert name in graph(), name


def test_everything_the_comma_imports_is_under_jetlink():
  for name, (f, _, _) in graph().items():
    assert f.is_relative_to(PACKAGE), (name, f)


def test_nothing_it_imports_is_from_the_repo_outside_jetlink():
  # scripts/, tests/ and the rest are not in the payload
  stray = {name: sorted(rest) for name, (_, _, theirs) in graph().items()
           if (rest := theirs - OUTSIDE - HOST.get(name, set()))}
  assert stray == {}


def test_only_root_finds_a_file_by_its_own_location():
  # a path off __file__ could reach past jetlink/; root.py's is its script, next to it
  located = {name for name, (_, tree, _) in graph().items()
             if any(isinstance(node, ast.Name) and node.id == '__file__' for node in ast.walk(tree))}
  assert located == {'jetlink.comma.root'}


def test_the_root_script_is_in_the_package():
  from jetlink.comma import root
  assert root.SCRIPT.resolve().is_relative_to(PACKAGE)
  assert root.SCRIPT.is_file()


def test_a_wheel_carries_the_root_script():
  tomllib = pytest.importorskip('tomllib')  # 3.11 and later
  config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
  assert config['tool']['setuptools']['package-data']['jetlink.comma'] == ['jetlink-root.sh']


CHECK = '''
import importlib, json, sys
sys.path.insert(0, sys.argv[1])
names = json.loads(sys.argv[2])
for name in names:
  importlib.import_module(name)
from jetlink.comma import root
print(json.dumps({'files': sorted({m.__file__ for n, m in sys.modules.items() if n.split('.')[0] == 'jetlink'}),
                  'script': str(root.SCRIPT), 'script_exists': root.SCRIPT.is_file()}))
'''


def test_the_payload_alone_runs_the_comma(tmp_path):
  for item in PAYLOAD:
    src = ROOT / item
    if src.is_dir():
      shutil.copytree(src, tmp_path / item, ignore=shutil.ignore_patterns('__pycache__'))
    else:
      shutil.copy(src, tmp_path / item)
  # every module the walk found is in the copy
  assert [n for n, (f, _, _) in graph().items() if not (tmp_path / f.relative_to(ROOT)).is_file()] == []

  names = [*ENTRIES, 'jetlink.comma.owner', 'jetlink.comma.root']
  # -I: no PYTHONPATH, no user site and no current directory, so only the copy
  # and the interpreter's own packages (numpy) are there
  out = subprocess.run([sys.executable, '-I', '-c', CHECK, str(tmp_path), json.dumps(names)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
  assert out.returncode == 0, out.stderr
  found = json.loads(out.stdout)
  copy = tmp_path.resolve()
  assert [f for f in found['files'] if not Path(f).resolve().is_relative_to(copy / 'jetlink')] == []
  assert Path(found['script']).resolve().is_relative_to(copy / 'jetlink')
  assert found['script_exists']
