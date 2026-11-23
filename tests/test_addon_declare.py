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
def vasm(tmp_path, monkeypatch):
  monkeypatch.setattr(A, 'REPORT_DIR', tmp_path)
  model = tmp_path / 'v_asm.onnx'
  model.write_bytes(b'onnx')
  return A.Addon('vasm', (0xFF, 0x56, 0x41), setting='ForkVasm', models=(('V-ASM', model, hashlib.sha256(b'onnx').hexdigest()),))


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


def test_model_wraps_any_runtime_and_follows_the_setting(vasm, params):
  calls = {}
  with mock.patch('jetlink.addons.link.host_model', lambda local, fn, sha, outputs, **kw: calls.update(local=local, fn=fn, **kw) or 'wrapped'):
    assert vasm.model(len, 'ab' * 32, [('out', (1, 3))], params=params) == 'wrapped'
  assert calls['local'] is len and calls['fn'] == vasm.function and calls['scaled_u8'] is False
  assert calls['enabled']() is False


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
  assert signs.state(None, None) == {'host': None, 'preparing': '', 'last_check': '', 'last_check_at': None}
  assert signs.state({'device': 'orin'}, None)['host'] == 'unsupported'
  assert signs.state({'addons': {'serving': {'ff/53/4c': ['a' * 64, 'b' * 64]}}}, None)['host'] == 'ready'
  assert signs.state({'addons': {'serving': {'ff/53/4c': ['a' * 64]}}}, None)['host'] == 'missing'
  assert signs.state({'addons': {'serving': {'ff/53/4c': ['a' * 64]}, 'failed': ['b' * 64]}}, None)['host'] == 'failed'
  # ready only on this add-on's own interface
  both = {'addons': {'serving': {'ff/53/4c': ['a' * 64], 'ff/56/41': ['b' * 64]}}}
  assert signs.state(both, None)['host'] == 'missing'
  both['addons']['serving']['ff/53/4c'].append('b' * 64)
  assert signs.state(both, None)['host'] == 'ready'
  # the last provisioning run: each of its models it sent, else the run's result
  run = {'at': 1000.0, 'result': 'no JetLink computer answered within 20 s', 'models': {}}
  assert signs.state(None, None, run)['last_check'] == 'no JetLink computer answered within 20 s'
  assert signs.state(None, None, run)['last_check_at'] == 1000.0
  run['models'] = {'ff/53/4c': {'b' * 64: {'name': 'Stop', 'state': 'failed', 'detail': 'too big'}, 'a' * 64: {'name': 'Sign', 'state': 'ready'}}}
  assert signs.state(None, None, run)['last_check'] == "Sign sent and built; Stop JetLink can't run it: too big"
  assert vasm.state(None, None, run)['last_check'] == run['result']
  assert signs.state(None, None, None) == {**signs.state(None, None), 'last_check': '', 'last_check_at': None}
  progress = {'msg': 'Stop: build 40%', 'code': 'addon_build', 'args': {'name': 'Stop', 'interface': 'ff/53/4c', 'sha256': 'b' * 64}}
  assert signs.state(None, progress)['preparing'] == 'Stop: build 40%'
  assert vasm.state(None, progress)['preparing'] == ''
  assert signs.state(None, {**progress, 'code': 'build'})['preparing'] == ''


def test_a_model_is_its_interface_and_sha256_not_its_name(vasm, tmp_path):
  # one file on two add-ons' interfaces, and one display name on both
  sha = vasm.models[0][2]
  other = A.Addon('other', (0xFF, 0x53, 0x4C), models=(('V-ASM', vasm.models[0][1], sha),))
  run = {'at': 1000.0, 'result': 'some add-on models are left for the next park',
         'models': {'ff/56/41': {sha: {'name': 'V-ASM', 'state': 'ready'}},
                    'ff/53/4c': {sha: {'name': 'V-ASM', 'state': 'left', 'detail': 'link lost'}}}}
  assert vasm.state(None, None, run)['last_check'] == 'V-ASM sent and built'
  assert other.state(None, None, run)['last_check'] == 'V-ASM left for the next park: link lost'
  progress = {'msg': 'V-ASM: build 40%', 'code': 'addon_build', 'args': {'name': 'V-ASM', 'interface': 'ff/53/4c', 'sha256': sha}}
  assert (vasm.state(None, progress)['preparing'], other.state(None, progress)['preparing']) == ('', 'V-ASM: build 40%')


def test_the_owner_can_import_it():
  # the owner reads ADDONS off the adapter, and stays at the standard library
  import subprocess
  import sys
  code = ("import sys, jetlink.addons.addon; jetlink.addons.addon.Addon('vasm', (0xFF, 0x56, 0x41)).present(); "
          "print(sorted(m for m in ('numpy', 'jetlink.addons.link') if m in sys.modules))")
  out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                       env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
  assert out.stdout.strip() == '[]'
