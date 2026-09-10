"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The backend seam as the engine host drives it, with no runtime installed:
selection, the hello, and an artifact the backend cannot load.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from jetlink import protocol as P
from jetlink.client import JetlinkClient
from jetlink.server import backends
from jetlink.server.backends.base import IO, infer, input_shapes, load_inputs
from jetlink.server.cache import EngineCache
from jetlink.server.session import EngineHost, Session
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError
from jetlink.transport.tcp import TcpTransport
from tests.fake_backend import FakeBackend, FakeEngine

SHAPES = {
  'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
  'action_t': (1, 2), 'features_buffer': (1, 32, 32, 512),
}


def make_spec(sha='d' * 64, nbytes=1000) -> ModelSpec:
  return ModelSpec(sha256=sha, nbytes=nbytes, frame_skip=4, input_shapes=SHAPES,
                   output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450), 'plan': slice(917, 1907)},
                   checkpoint=None)


# -- selection ----------------------------------------------------------------

class TestSelect:
  def test_an_unknown_name_is_refused(self):
    with pytest.raises(ValueError, match='unknown backend'):
      backends.select('cuda')

  def test_nothing_installed_says_what_to_install(self, monkeypatch):
    monkeypatch.setattr(backends, '_candidates', lambda device: [])
    with pytest.raises(RuntimeError, match=r'jetlink\[trt\]'):
      backends.select('auto')

  def test_auto_moves_past_a_backend_that_will_not_come_up(self, monkeypatch, caplog):
    made = []

    def make(name, device):
      made.append(name)
      if name == 'trt':
        raise RuntimeError('no CUDA device')
      return FakeBackend(version=name)

    monkeypatch.setattr(backends, '_candidates', lambda device: [('trt', device), ('tinygrad', device)])
    monkeypatch.setattr(backends, '_make', make)
    picked = backends.select('auto', 'METAL')
    assert picked.version == 'tinygrad'
    assert made == ['trt', 'tinygrad']
    assert 'no CUDA device' in caplog.text

  def test_auto_with_nothing_working_reports_every_reason(self, monkeypatch):
    def make(name, device):
      raise RuntimeError(f'{name} is broken')

    monkeypatch.setattr(backends, '_candidates', lambda device: [('trt', device), ('ort', device)])
    monkeypatch.setattr(backends, '_make', make)
    with pytest.raises(RuntimeError, match='trt is broken') as e:
      backends.select('auto')
    assert 'ort is broken' in str(e.value)

  def test_available_lists_installed_runtimes_only(self, monkeypatch):
    monkeypatch.setattr(backends.sys, 'platform', 'linux')
    monkeypatch.setattr(backends, '_importable', lambda m: m in ('tinygrad',))
    assert backends.available() == ['tinygrad']
    monkeypatch.setattr(backends, '_importable', lambda m: m in ('tensorrt', 'cuda.bindings', 'onnxruntime'))
    assert backends.available() == ['trt', 'ort']
    monkeypatch.setattr(backends, '_importable', lambda m: m in ('onnxruntime', 'tinygrad'))
    assert backends.available() == ['tinygrad', 'ort']
    monkeypatch.setattr(backends.sys, 'platform', 'darwin')
    assert backends.available() == ['ort', 'tinygrad']

  def test_a_mac_tries_coreml_by_name_then_tinygrad(self, monkeypatch):
    """An onnxruntime without the CoreML provider must not serve off the CPU
    on auto; tinygrad on Metal is the next best, not a 600 ms frame."""
    monkeypatch.setattr(backends.sys, 'platform', 'darwin')
    monkeypatch.setattr(backends, '_importable', lambda m: m in ('onnxruntime', 'tinygrad'))
    assert backends._candidates('auto') == [('ort', 'coreml'), ('tinygrad', 'auto')]
    assert backends._candidates('METAL') == [('ort', 'METAL'), ('tinygrad', 'METAL')]
    monkeypatch.setattr(backends.sys, 'platform', 'linux')
    monkeypatch.setattr(backends, '_importable', lambda m: m in ('tensorrt', 'cuda', 'onnxruntime', 'tinygrad'))
    assert backends._candidates('auto') == [('trt', 'auto'), ('tinygrad', 'auto'), ('ort', 'auto')]


# -- the base helpers ---------------------------------------------------------

def test_load_inputs_casts_and_checks_size():
  spec = make_spec()
  engine = FakeEngine(spec)
  load_inputs(engine, {'img': np.full(SHAPES['img'], 7, np.uint8)})
  assert engine.host_input('img').dtype == np.float16
  assert float(engine.host_input('img')[0, 0, 0, 0]) == 7.0
  with pytest.raises(KeyError, match='no input'):
    load_inputs(engine, {'nope': np.zeros(1)})
  with pytest.raises(ValueError, match='elements'):
    load_inputs(engine, {'img': np.zeros(3)})
  out = infer(engine, {'action_t': np.ones((1, 2), np.float32)})
  assert 'outputs' in out
  assert input_shapes(engine)['action_t'] == (1, 2)
  assert IO('x', (1,), np.dtype(np.float16)).shape == (1,)


# -- the engine host end to end -----------------------------------------------

@pytest.fixture
def linked(tmp_path):
  """A real client and Session over a socket, with a FakeBackend behind the host."""
  spec = make_spec()
  onnx = tmp_path / 'model.onnx'
  onnx.write_bytes(b'not really onnx' + bytes(spec.nbytes - 15))
  backend = FakeBackend(spec)
  host = EngineHost(EngineCache(tmp_path / 'cache', backend))
  host._derive_spec = lambda path, frame_skip: spec   # no parser off a real model

  srv = TcpTransport.listen('127.0.0.1', 0)
  port = srv.getsockname()[1]
  client_t = TcpTransport.connect('127.0.0.1', port)
  server_t, _ = TcpTransport.accept(srv)
  srv.close()
  session = Session(server_t, host)
  thread = threading.Thread(target=session.serve_forever, daemon=True)
  thread.start()
  client = JetlinkClient(client_t, deadline=10.0)
  yield client, host, backend, spec, onnx
  client.close()
  server_t.close()
  thread.join(1.0)
  host.close()


def _sha_of(spec, onnx, monkeypatch):
  # The upload path hashes what arrived; make the fake model hash to the spec's identity.
  from jetlink.server import session as session_module
  monkeypatch.setattr(session_module, 'sha256_file', lambda path: (spec.sha256, Path(path).stat().st_size))


def test_upload_build_load_and_run_through_the_seam(linked, monkeypatch):
  client, host, backend, spec, onnx = linked
  _sha_of(spec, onnx, monkeypatch)
  stages = []
  got = client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=onnx,
                             progress=lambda s, f, m: stages.append(s), build_timeout=10.0)
  assert got.sha256 == spec.sha256
  assert [p.name for p in backend.builds] == [f"{spec.sha256[:16]}.fake0.1.test.fake"]
  assert backend.loads == backend.builds
  assert backend.engines[-1].warmed == 1
  assert 'upload' in stages and 'build' in stages and 'load' in stages
  # the sidecar carries the spec, so the next process loads without the ONNX
  meta = json.loads(backend.builds[0].with_suffix('.json').read_text())
  assert meta['backend'] == 'fake' and meta['spec']['sha256'] == spec.sha256
  assert host.cache.inventory() == [spec.sha256]

  out = client.infer(np.zeros(spec.warped_shape, np.uint8), np.zeros(spec.packed_nelem, np.float32))
  assert out.shape == (spec.output_nelem,)
  assert backend.engines[-1].calls == 2   # warm, then the frame


def test_hello_names_the_backend_and_keeps_trt_version_only_for_tensorrt(linked):
  client, host, backend, spec, onnx = linked
  hello = client.hello(timeout=5)
  assert hello['backend'] == 'fake'
  assert hello['runtime_version'] == '0.1'
  assert hello['device'] == 'test'
  assert 'trt_version' not in hello
  backend.name = 'trt'
  hello = client.hello(timeout=5)
  assert hello['trt_version'] == '0.1'


def test_an_invalid_artifact_is_rebuilt_from_the_model_on_disk(linked, monkeypatch, caplog):
  """A pickle from an older tinygrad must heal, not fail every connect."""
  client, host, backend, spec, onnx = linked
  _sha_of(spec, onnx, monkeypatch)
  client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=onnx, build_timeout=10.0)
  host._unload()
  with host.lock:
    host.loaded = None

  backend.invalid_loads = 1
  client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  assert len(backend.builds) == 2, 'the invalid artifact was not rebuilt'
  assert len(backend.loads) == 3   # first load, the invalid one, the load after the rebuild
  assert 'discarding' in caplog.text
  assert host.cache.entry(spec.sha256).exists


def test_an_invalid_artifact_with_no_model_asks_for_an_upload(linked, monkeypatch):
  client, host, backend, spec, onnx = linked
  entry = host.cache.entry(spec.sha256)
  entry.path.write_text('stale')
  entry.write_meta({'spec': spec.to_dict()})
  backend.invalid_loads = 1

  with pytest.raises(LinkError, match='not on disk'):
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  assert not entry.exists, 'the invalid artifact was left for the next connect to trip on'
  # The next ask is answered from the disk, not the failed job: EngineMissing
  # is what makes the comma clear JetlinkEngineReady and provision again.
  from jetlink.client import EngineMissing
  with pytest.raises(EngineMissing):
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  assert host.status(spec.sha256, spec.frame_skip)['state'] == 'need_upload'


def test_an_invalid_artifact_twice_running_fails_rather_than_loops(linked, monkeypatch):
  client, host, backend, spec, onnx = linked
  _sha_of(spec, onnx, monkeypatch)
  client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=onnx, build_timeout=10.0)
  host._unload()
  backend.invalid_loads = 2
  with pytest.raises(LinkError, match='made invalid'):
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  assert len(backend.builds) == 2


def test_a_load_failure_that_is_not_the_artifacts_fault_keeps_it(linked, monkeypatch):
  """TensorRT out of memory, a device fault: the plan stays, the job fails."""
  client, host, backend, spec, onnx = linked
  _sha_of(spec, onnx, monkeypatch)
  client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=onnx, build_timeout=10.0)
  host._unload()

  def boom(artifact, report=None):
    raise MemoryError('device out of memory')
  monkeypatch.setattr(backend, 'load', boom)
  with pytest.raises(LinkError, match='out of memory'):
    client.ensure_engine(spec.sha256, spec.nbytes, onnx_path=None, build_timeout=10.0)
  assert host.cache.entry(spec.sha256).exists
  assert len(backend.builds) == 1


def test_shape_check_names_a_missing_input(linked):
  from jetlink.server.session import _check_shapes
  client, host, backend, spec, onnx = linked
  engine = FakeEngine(spec)
  del engine.inputs['action_t']
  with pytest.raises(ValueError, match="action_t"):
    _check_shapes(engine, spec)
  engine = FakeEngine(spec)
  engine.inputs['img'] = IO('img', (1, 1), np.dtype(np.float16))
  with pytest.raises(ValueError, match='input img'):
    _check_shapes(engine, spec)


def test_the_ping_does_not_need_an_engine(linked):
  client, *_ = linked
  assert client.ping(timeout=5) < 5.0
  assert client.state(timeout=5)['engine_state'] == 'none'
  assert P.VERSION == client.hello(timeout=5)['protocol']


def test_the_ort_load_ticker_logs_once_a_minute(caplog):
  """A CoreML load ticks every 5 s for nine minutes: a hundred log lines, but
  the progress a client draws has to move on every one of them."""
  import logging

  from jetlink.server.backends.ort import load_ticker

  reports = []
  tick = load_ticker(lambda stage, frac, msg: reports.append((stage, frac, msg)),
                     lambda elapsed: f'{elapsed / 60:.0f} min elapsed')
  elapsed = [5.0, 10.2, 30.5, 55.1, 60.3, 65.4, 119.8, 120.6, 180.9]
  with caplog.at_level(logging.DEBUG, logger='jetlink.ort'):
    for e in elapsed:
      tick(e)

  info = [r for r in caplog.records if r.levelno == logging.INFO]
  assert len(info) == 4, 'the first tick and one a minute after it'
  assert 'still creating the onnxruntime sessions, 5 s' in info[0].getMessage()
  assert [round(float(r.getMessage().split(', ')[1].split(' ')[0])) for r in info] == [5, 60, 121, 181]
  assert len([r for r in caplog.records if r.levelno == logging.DEBUG]) == len(elapsed) - 4
  # Every tick still moves the progress, whatever the log did.
  assert len(reports) == len(elapsed)
  assert reports[0] == ('load', 0.0, '0 min elapsed')
