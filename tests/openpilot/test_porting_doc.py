"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

docs/porting.md, the lines a fork copies, against the code they call. Each
python snippet parses, and every name it takes from jetlink is there: the
hooks it calls, what it overrides on Stock, what it reads off the status and
the swap flags, what it imports, and the keywords it passes. The fork's own
fallback answers every hook a link that never runs reaches, as Off does.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

from jetlink.openpilot import SHUTDOWN_TIMEOUT, Flags, Status
from jetlink.openpilot.hooks import Hooks, Off, SwapTracker
from jetlink.openpilot.stock import Stock
from jetlink.openpilot.testing import check_fallback
from tests.test_comma_payload import PAYLOAD

DOC = Path(__file__).resolve().parents[2] / 'docs' / 'porting.md'
TEXT = DOC.read_text()


def snippets() -> list[ast.Module]:
  """Each python fence, parsed whole, or paragraph by paragraph where it
  mixes a module's top level with lines from inside a function."""
  trees = []
  for block in re.findall(r'^```python\n(.*?)^```', TEXT, re.M | re.S):
    try:
      trees.append(ast.parse(textwrap.dedent(block)))
    except SyntaxError:
      trees += [ast.parse(textwrap.dedent(part), str(DOC)) for part in re.split(r'\n\s*\n', block) if part.strip()]
  return trees


TREES = snippets()
NODES = [node for tree in TREES for node in ast.walk(tree)]


def public(cls) -> set[str]:
  return {name for name in dir(cls) if not name.startswith('_')}


def hooks_used() -> set[str]:
  """Every HOOKS.<name> and hooks().<name> on the page, in its prose too:
  `HOOKS.state(model)` is a line a fork writes in its frame loop."""
  found = set(re.findall(r'\bHOOKS\.(\w+)', TEXT))
  for node in NODES:
    if isinstance(node, ast.Attribute):
      value = node.value
      if (isinstance(value, ast.Name) and value.id == 'HOOKS') or \
         (isinstance(value, ast.Call) and getattr(value.func, 'id', getattr(value.func, 'attr', None)) == 'hooks'):
        found.add(node.attr)
  return found


def read_off(name: str) -> set[str]:
  """The attributes the page reads off the variable `name`."""
  return {n.attr for n in NODES if isinstance(n, ast.Attribute) and ast.unparse(n.value) == name}


def classes(name: str) -> list[ast.ClassDef]:
  return [n for n in NODES if isinstance(n, ast.ClassDef) and n.name == name]


def defined(cls: ast.ClassDef) -> dict[str, list[str] | None]:
  """A class body's names: a method's parameters without self, None for data."""
  names: dict[str, list[str] | None] = {}
  for item in cls.body:
    if isinstance(item, ast.FunctionDef):
      names[item.name] = [a.arg for a in item.args.args[1:]]
    elif isinstance(item, ast.Assign):
      names.update((t.id, None) for t in item.targets if isinstance(t, ast.Name))
  return names


def parameters(cls, name: str) -> list[str]:
  return list(inspect.signature(getattr(cls, name)).parameters)[1:]


# the hooks a link that never runs does not reach: each only inside an `if`
# whose test has this in it
GATED = {'attach': 'HOOKS.prepare('}


def gates(name: str) -> list[list[str]]:
  """For each HOOKS.<name> on the page, the tests of the ifs around it."""
  found = []
  for tree in TREES:
    def walk(node, tests):
      if isinstance(node, ast.Attribute) and ast.unparse(node.value) == 'HOOKS' and node.attr == name:
        found.append(tests)
      inner = tests + [ast.unparse(node.test)] if isinstance(node, ast.If) else tests
      for child in ast.iter_child_nodes(node):
        walk(child, inner if child is not getattr(node, 'test', None) else tests)
    walk(tree, [])
  return found


def test_the_page_has_its_snippets():
  assert len(TREES) >= 10
  assert {'prepare', 'attach', 'handed_over', 'state', 'shutdown_step', 'reason', 'status', 'swap_events',
          'should_run', 'runtime'} <= hooks_used() | set(re.findall(r'HOOKS\.(\w+)\(', TEXT))


def test_every_hook_it_calls_is_a_hook():
  # in the snippets and in the text around them
  used = hooks_used() | set(re.findall(r'HOOKS\.(\w+)\(', TEXT))
  assert used - public(Hooks) == set()


def test_the_hooks_past_a_link_that_runs_are_gated():
  for name, gate in GATED.items():
    assert gates(name), name
    assert all(any(gate in test for test in tests) for tests in gates(name)), name


def test_the_fallback_answers_every_hook_a_link_that_never_runs_reaches_as_off_does():
  [fallback] = classes('_NoJetlink')
  methods = {k: v for k, v in defined(fallback).items() if k != '__init__'}
  reached = hooks_used() - GATED.keys()
  assert reached == methods.keys()
  for name, params in methods.items():
    assert params == parameters(Hooks, name), name
  # run: the adapter module's top level and the fallback, over a params store of its own
  module = next(t for t in TREES if any(isinstance(n, ast.FunctionDef) and n.name == 'hooks' for n in t.body))
  space: dict = {}
  exec(compile(ast.Module(body=module.body + [fallback], type_ignores=[]), str(DOC), 'exec'), space)
  with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {'PARAMS_ROOT': root}):
    assert check_fallback(space['_NoJetlink']('jetlink is missing'), reached, link_on=space['link_on']) == []
  # and the page's own test of a fork's fallback names exactly these
  [listed] = [ast.literal_eval(n.value) for n in NODES if isinstance(n, ast.Assign)
              and [ast.unparse(x) for x in n.targets] == ['hooks'] and isinstance(n.value, ast.Tuple)]
  assert set(listed) == reached


def test_an_adapter_overrides_only_what_stock_has():
  subclasses = [n for n in NODES if isinstance(n, ast.ClassDef) and any(ast.unparse(b) == 'Stock' for b in n.bases)]
  assert subclasses
  for cls in subclasses:
    assert [name for name in defined(cls) if not hasattr(Stock, name)] == [], cls.name
    # what its docstring says to override: KEYS, engaged()
    named = re.findall(r'\b([A-Z][A-Z_]{2,})\b|\b(\w+)\(\)', ast.get_docstring(cls) or '')
    assert [n for pair in named for n in pair if n and not hasattr(Stock, n)] == [], cls.name
  for name in re.findall(r'`Stock\.(\w+)`', TEXT):
    assert hasattr(Stock, name), name


def test_the_hook_lines_call_what_the_adapter_module_defines():
  module = next(t for t in TREES if any(isinstance(n, ast.FunctionDef) and n.name == 'hooks' for n in t.body))
  names = {n.name for n in module.body if isinstance(n, ast.FunctionDef)} | \
          {t.id for n in module.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
  names |= {n.name for n in NODES if isinstance(n, ast.ClassDef)}
  assert read_off('jetlink_adapter') - names == set()


def test_what_it_reads_is_there():
  assert read_off('status') - public(Status) == set()
  assert read_off('flags') - set(Flags._fields) == set()
  assert read_off('self.jetlink_swaps') - public(SwapTracker) == set()
  for attr in re.findall(r'`status\.(\w+)', TEXT):
    assert hasattr(Status, attr), attr


def test_its_imports_and_keywords_are_there():
  for node in NODES:
    if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('jetlink'):
      module = importlib.import_module(node.module)
      assert [a.name for a in node.names if not hasattr(module, a.name)] == [], node.module
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ('Hooks', 'Off', 'check_adapter', 'check_seam'):
      target = {'Hooks': Hooks, 'Off': Off}.get(ast.unparse(node.func)) or \
        getattr(importlib.import_module('jetlink.openpilot.testing'), ast.unparse(node.func))
      accepted = inspect.signature(target).parameters
      assert [k.arg for k in node.keywords if k.arg not in accepted] == [], ast.unparse(node.func)


def test_the_shutdown_cap_is_the_timeout_and_five_seconds():
  [cap] = re.findall(r'JETLINK_SHUTDOWN_CAP_S = ([\d.]+)', TEXT)
  assert float(cap) == SHUTDOWN_TIMEOUT + 5
  assert f'of {SHUTDOWN_TIMEOUT:g} s plus 5 s' in TEXT


def test_its_keys_are_stocks():
  declared = re.findall(r'^\s+\{"(\w+)", \{', TEXT, re.M)
  keys = Stock.KEYS
  assert {keys.link, keys.progress, keys.spec, keys.charge_phone} <= set(declared)
  assert keys.offroad not in declared   # openpilot's own


def test_it_names_the_payload():
  for path in PAYLOAD:
    assert f'`{path}/`' in TEXT or f'`{path}`' in TEXT, path
