"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink.addons.addon: a fork's add-on declared once, as the adapter, the
model process and a panel use it.
"""
import hashlib
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from jetlink.addons import addon as A
from jetlink.openpilot.interface import Keys
from jetlink.openpilot.settings import FileParams
from jetlink.openpilot.stock import Stock


@pytest.fixture
def params(tmp_path):
  store = tmp_path / 'params'
  store.mkdir()
  return FileParams(store)


@pytest.fixture
def vasm(tmp_path):
  model = tmp_path / 'v_asm.onnx'
  model.write_bytes(b'onnx')
  return A.Addon('vasm', (0xFF, 0x56, 0x41), setting='ForkVasm', models=(('V-ASM', model, hashlib.sha256(b'onnx').hexdigest()),),
                 report_path=tmp_path / 'vasm.json')


def test_its_models_go_to_the_jetson_only_while_it_is_on(vasm, params):
  assert vasm.addon_models(params) == ()
  params.path('ForkVasm').write_bytes(b'1')
  (model,) = vasm.addon_models(params)
  assert (model.function, model.name, model.sha256) == (vasm.function, 'V-ASM', vasm.models[0][2])
  assert A.Addon('x', (0xFF, 1, 2), models=vasm.models).addon_models(params), 'no setting is always on'


def test_the_stock_adapter_takes_its_interface_models_and_setting(vasm, params):
  class Adapter(Stock):
    KEYS = Keys(link='L', offroad='O', progress='P', spec='S', addons=('Other',))
    ADDONS = (vasm,)

  assert Adapter.gadget_functions(params) == (vasm.function,)
  params.path('ForkVasm').write_bytes(b'1')
  assert [m.name for m in Adapter.addon_models(params)] == ['V-ASM']
  assert Adapter().keys.addons == ('Other', 'ForkVasm')
  assert Stock().keys is Stock.KEYS


def test_the_setting_is_read_live_and_cheaply(params):
  on = A.live_setting(params, 'ForkVasm', refresh_s=0.05)
  assert on() is False
  params.path('ForkVasm').write_bytes(b'1')
  assert on() is False, 'read again only after refresh_s'
  time.sleep(0.06)
  assert on() is True
  assert A.live_setting(params, None)() is True


def test_wrap_reports_the_link_and_follows_the_setting(vasm, params):
  calls = {}
  with mock.patch('jetlink.addons.link.with_host', lambda net, fn, sha, outputs, **kw: calls.update(fn=fn, sha=sha, **kw) or 'wrapped'):
    assert vasm.wrap('net', 'ab' * 32, [('out', (1, 3))], params=params, scaled_u8=True) == 'wrapped'
  assert calls['fn'] == vasm.function and calls['scaled_u8'] is True
  assert calls['enabled']() is False
  calls['report']({'connected': True})
  assert vasm.report() == {'connected': True}
  assert vasm.report(now=time.time() + A.REPORT_FRESH_S + 1) is None
  assert not [p for p in vasm.report_file.parent.iterdir() if p.name.startswith('.')], 'no temporary left'


def test_the_panel_state_is_the_worst_of_its_models(vasm, tmp_path):
  stop = tmp_path / 'stop.onnx'
  stop.write_bytes(b'stop')
  signs = A.Addon('signs', (0xFF, 0x53, 0x4C), models=(('Sign', vasm.models[0][1], 'a' * 64), ('Stop', stop, 'b' * 64)))
  assert signs.state(None, None) == {'jetson': None, 'preparing': ''}
  assert signs.state({'device': 'orin'}, None)['jetson'] == 'unsupported'
  assert signs.state({'addons': {'ready': ['a' * 64, 'b' * 64]}}, None)['jetson'] == 'ready'
  assert signs.state({'addons': {'ready': ['a' * 64]}}, None)['jetson'] == 'missing'
  assert signs.state({'addons': {'ready': ['a' * 64], 'failed': ['b' * 64]}}, None)['jetson'] == 'failed'
  progress = {'msg': 'Stop: build 40%', 'code': 'addon_build', 'args': {'name': 'Stop'}}
  assert signs.state(None, progress)['preparing'] == 'Stop: build 40%'
  assert vasm.state(None, progress)['preparing'] == ''
  assert signs.state(None, {**progress, 'code': 'build'})['preparing'] == ''


def test_the_owner_can_import_it():
  # the owner reads ADDONS off the adapter, and stays at the standard library
  import subprocess
  import sys
  code = "import sys, jetlink.addons.addon; print(sorted(m for m in ('numpy', 'jetlink.addons.link') if m in sys.modules))"
  out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                       env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
  assert out.stdout.strip() == '[]'
