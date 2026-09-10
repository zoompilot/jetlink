"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The control channel, against a real EngineHost and a stand-in registry.

The socket, the framing, the event order on connect, the command dispatch and
the download bookkeeping are the code that will run under the Mac app. Only the
backend (FakeBackend) and the network side of the registry are faked; the
registry stand-in implements the API in plans/macos-app/01-contracts.md
section 5, so this file runs whether or not jetlink.registry is merged yet.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from jetlink import protocol as P
from jetlink.queues import PolicyQueues
from jetlink.server.cache import EngineCache
from jetlink.server.control import ControlServer
from jetlink.server.session import EngineHost, Loaded, Request, Session
from jetlink.spec import ModelSpec
from jetlink.transport.base import LinkError, Message
from tests.fake_backend import FakeBackend, FakeEngine

SHA = 'b' * 64
OTHER = 'c' * 64
REF = 'f877d7a0ccc3cce943c76e285214c020cd65c899'

SHAPES = {
  'img': (1, 12, 8, 16), 'big_img': (1, 12, 8, 16),
  'desire_pulse': (1, 33, 8), 'traffic_convention': (1, 2),
  'action_t': (1, 2), 'features_buffer': (1, 32, 4, 8),
}


def make_spec(sha256: str = SHA, nbytes: int = 512) -> ModelSpec:
  return ModelSpec(sha256=sha256, nbytes=nbytes, frame_skip=4, input_shapes=SHAPES,
                   output_shapes={'outputs': (1, 64)},
                   output_slices={'plan': slice(0, 16), 'hidden_state': slice(32, 64)},
                   checkpoint=None)


# -- the registry stand-in ---------------------------------------------------


@dataclass(frozen=True)
class Pointer:
  oid: str
  size: int


@dataclass(frozen=True)
class LocalModel:
  sha256: str
  bytes: int
  name: str
  added_at: float


class FakeRegistry:
  """01-contracts.md section 5, with the network replaced by dictionaries."""

  def __init__(self, root: Path):
    self.root = Path(root)
    self.models = self.root / 'models'
    self.models.mkdir(parents=True, exist_ok=True)
    self.catalog_path = self.root / 'registry' / 'catalog.json'
    self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    self.catalog_path.write_text('{}')   # the tests that want one on disk are the default
    self.pointers: dict[str, Pointer] = {}
    self.resolvable: dict[str, Pointer] = {}
    self.names: dict[str, str] = {SHA: 'Test Model v1'}
    self.entries: list[dict] = []
    self.refreshes = 0
    self.resolved: list[list[str]] = []
    self.removals: list[tuple] = []
    self.fetch_impl = None
    self.import_impl = None

  # catalog
  def catalog(self, refresh: bool = False, max_age: float = 3600.0, opener=None) -> dict:
    if refresh:
      self.refreshes += 1
    models = []
    for e in self.entries:
      pointer = self.pointers.get(e['ref'])
      models.append({**e, 'sha256': pointer.oid if pointer else None,
                     'bytes': pointer.size if pointer else None})
    return {'fetched_at': 1757440000.0, 'url': 'https://example.invalid/catalog.json',
            'default_ref': REF, 'error': None, 'models': models}

  def resolve(self, ref: str, opener=None) -> Pointer:
    if ref in self.pointers:
      return self.pointers[ref]
    if ref not in self.resolvable:
      raise KeyError(f'no pointer for {ref}')
    self.pointers[ref] = self.resolvable[ref]
    return self.pointers[ref]

  def resolve_missing(self, refs, workers: int = 8, opener=None) -> dict:
    self.resolved.append(list(refs))
    out = {}
    for ref in refs:
      try:
        out[ref] = self.resolve(ref)
      except KeyError as e:
        out[ref] = e
    return out

  def name_for(self, sha256: str):
    for ref, pointer in self.pointers.items():
      if pointer.oid == sha256:
        return self.names.get(sha256), ref
    return self.names.get(sha256), None

  # models on disk
  def model_path(self, sha256: str) -> Path:
    return self.models / f'{sha256[:16]}.onnx'

  def fetch(self, ref_or_sha256: str, progress=None, should_stop=None, opener=None) -> Path:
    return self.fetch_impl(ref_or_sha256, progress, should_stop)

  def import_model(self, path: Path, name=None, progress=None, should_stop=None) -> LocalModel:
    return self.import_impl(path, name, progress)

  def local_models(self):
    return []

  # inventory and removal
  def inventory(self, cache=None) -> dict:
    models = []
    for p in sorted(self.models.glob('*.onnx')):
      sha256 = next((s for s in self.names if s.startswith(p.stem)), p.stem + '0' * (64 - len(p.stem)))
      name, ref = self.name_for(sha256)
      models.append({'sha256': sha256, 'bytes': p.stat().st_size, 'path': str(p),
                     'name': name, 'ref': ref})
    artifacts = []
    if cache is not None:
      for meta_path in sorted(cache.engines.glob('*.json')):
        try:
          meta = json.loads(meta_path.read_text())
          sha256 = meta['spec']['sha256']
        except (OSError, ValueError, KeyError, TypeError):
          continue
        entry = cache.entry(sha256)
        artifacts.append({
          'sha256': sha256, 'key': meta_path.stem, 'path': str(entry.path),
          'bytes': entry.path.stat().st_size if entry.path.exists() else 0,
          'backend': meta.get('backend', ''), 'runtime_version': meta.get('onnxruntime'),
          'device': '', 'built_at': None, 'build_seconds': None, 'checkpoint': None,
          'current': meta_path.stem == cache.key(sha256)})
    last = cache.last_loaded() if cache is not None else None
    return {'loaded': None, 'last_loaded': last[0] if last else None,
            'models': models, 'artifacts': artifacts,
            'disk': {'models_bytes': 0, 'engines_bytes': 0, 'free_bytes': 0}}

  def remove(self, sha256: str, artifacts: bool, model: bool) -> None:
    self.removals.append((sha256, artifacts, model))
    if model:
      self.model_path(sha256).unlink(missing_ok=True)
    if artifacts:
      for p in (self.root / 'engines').glob(f'{sha256[:16]}.*'):
        p.unlink(missing_ok=True)


# -- harness -----------------------------------------------------------------


class Client:
  """A control client: connect, read events, send commands."""

  def __init__(self, address: str, timeout: float = 5.0):
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.sock.settimeout(timeout)
    self.sock.connect(address)
    self.f = self.sock.makefile('rb')
    self.seen: list[dict] = []

  def read(self) -> dict:
    line = self.f.readline()
    assert line, 'the control server closed the connection'
    return json.loads(line)

  def send(self, **msg) -> None:
    self.sock.sendall(json.dumps(msg).encode() + b'\n')

  def send_raw(self, data: bytes) -> None:
    self.sock.sendall(data)

  def wait_for(self, event: str, timeout: float = 5.0, **match) -> dict:
    """The next event of this shape, read past or already read.

    Events arrive between a command and its reply, so waiting for the reply
    must not lose them; whatever does not match is kept for the next wait.
    """
    def hit(msg: dict) -> bool:
      return msg['event'] == event and all(msg.get(k) == v for k, v in match.items())

    for i, msg in enumerate(self.seen):
      if hit(msg):
        del self.seen[i]
        return msg
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      msg = self.read()
      if hit(msg):
        return msg
      self.seen.append(msg)
    raise AssertionError(f'no {event} event with {match}')

  def close(self) -> None:
    try:
      self.f.close()
    finally:
      self.sock.close()


@pytest.fixture
def bench(tmp_path_factory):
  """A host, a cache, a registry and a listening control server."""
  root = tmp_path_factory.mktemp('jl')
  spec = make_spec()
  cache = EngineCache(root, FakeBackend(spec))
  host = EngineHost(cache)
  host._derive_spec = lambda model_path, frame_skip: spec
  registry = FakeRegistry(root)
  server = ControlServer(str(root / 'c.sock'), host, cache, registry, info={
    'version': '0.2.0', 'python': '3.14.7', 'platform': 'darwin',
    'cache': str(root), 'transport': 'usb', 'port': None})
  server.start()
  clients: list[Client] = []

  def connect(drain: bool = True) -> Client:
    c = Client(server.address)
    clients.append(c)
    if drain:
      for _ in range(6):
        c.read()   # the on-connect burst; the tests below want what follows it
    return c

  yield SimpleNamespace(root=root, spec=spec, cache=cache, host=host, registry=registry,
                        server=server, connect=connect)
  for c in clients:
    c.close()
  server.close()
  host.close()


def write_model(bench, sha256: str = SHA) -> Path:
  """A model file the size the spec declares, so a request builds from it."""
  path = bench.cache.model_path(sha256)
  path.write_bytes(b'x' * bench.spec.nbytes)
  return path


# -- on connect --------------------------------------------------------------


def test_a_client_is_told_everything_before_it_asks(bench):
  c = bench.connect(drain=False)
  events = [c.read() for _ in range(6)]
  assert [e['event'] for e in events] == ['hello', 'server', 'link', 'engine', 'inventory', 'catalog']
  hello, server, link, engine, inventory, catalog = events
  assert hello['protocol'] == 1 and hello['pid'] > 0
  assert hello['version'] == '0.2.0' and hello['transport'] == 'usb' and hello['port'] is None
  assert server['state'] == 'serving' and server['backend'] == 'fake'
  assert link['state'] == 'waiting'
  assert engine['state'] == 'none' and engine['sha256'] is None and engine['load_only'] is False
  assert inventory['models'] == [] and inventory['artifacts'] == []
  assert catalog['models'] == []
  assert all(isinstance(e['t'], float) for e in events)


def test_status_resends_the_five_snapshots(bench):
  c = bench.connect()
  c.send(id=7, cmd='status')
  seen = []
  while True:
    msg = c.read()
    if msg['event'] == 'reply':
      assert msg['id'] == 7 and msg['ok'] is True
      break
    seen.append(msg['event'])
  assert seen == ['server', 'link', 'engine', 'inventory', 'catalog']


# -- prepare, unload, forget -------------------------------------------------


def test_prepare_builds_then_loads_and_shows_up_in_the_inventory(bench):
  write_model(bench)
  c = bench.connect()
  c.send(id=1, cmd='prepare', sha256=SHA)
  assert c.wait_for('engine', state='building')['sha256'] == SHA
  assert c.wait_for('engine', state='ready')['sha256'] == SHA
  inventory = c.wait_for('inventory')
  assert inventory['loaded'] == SHA
  assert [a['sha256'] for a in inventory['artifacts']] == [SHA]
  assert inventory['artifacts'][0]['current'] is True
  assert len(bench.cache.backend.builds) == 1

  # Loaded already: no second build, and the reply says so straight away.
  c.send(id=2, cmd='prepare', sha256=SHA)
  reply = c.wait_for('reply', id=2)
  assert reply['ok'] is True and reply['state'] == 'ready'
  assert len(bench.cache.backend.builds) == 1


def test_prepare_for_a_model_that_is_not_here_is_refused(bench):
  c = bench.connect()
  c.send(id=3, cmd='prepare', sha256=OTHER)
  reply = c.wait_for('reply', id=3)
  assert reply['ok'] is False and 'not downloaded' in reply['error']


def test_unload_and_forget(bench):
  write_model(bench)
  c = bench.connect()
  c.send(id=1, cmd='prepare', sha256=SHA)
  c.wait_for('engine', state='ready')
  assert bench.cache.last_loaded() is not None

  c.send(id=2, cmd='unload')
  assert c.wait_for('engine', state='none')['sha256'] is None
  assert bench.host.loaded_sha() is None

  c.seen.clear()                            # a build emits engine twice; do not match the old one
  c.send(id=3, cmd='prepare', sha256=SHA)   # loaded again, so forget has to unload first
  c.wait_for('engine', state='ready')
  c.send(id=4, cmd='forget', sha256=SHA, artifacts=True, model=True)
  reply = c.wait_for('reply', id=4)
  assert reply['ok'] is True, reply['error']
  assert bench.host.loaded_sha() is None
  assert bench.registry.removals == [(SHA, True, True)]
  assert not bench.cache.entry(SHA).exists
  assert not (bench.cache.root / 'last-loaded.json').exists()


def test_forget_is_refused_while_a_build_is_running(bench):
  from jetlink.server.session import Job
  bench.host.job = Job(SHA, load_only=False)
  c = bench.connect()
  c.send(id=1, cmd='forget', sha256=SHA, artifacts=True, model=False)
  reply = c.wait_for('reply', id=1)
  assert reply['ok'] is False and 'build for this model is running' in reply['error']


# -- downloads ---------------------------------------------------------------


def test_a_download_reports_progress_and_refreshes_the_inventory(bench):
  bench.registry.resolvable[REF] = Pointer(SHA, 4096)

  def fetch(ref_or_sha, progress, should_stop):
    for frac in (0.0, 0.5, 1.0):
      progress(frac)
      time.sleep(0.3)   # the events are throttled to four a second
    path = bench.registry.model_path(SHA)
    path.write_bytes(b'x' * 4096)
    return path

  bench.registry.fetch_impl = fetch
  c = bench.connect()
  c.send(id=1, cmd='download', ref=REF)
  assert c.wait_for('reply', id=1)['sha256'] == SHA
  assert c.wait_for('download', state='started')['total'] == 4096
  progress = c.wait_for('download', state='progress')
  assert 0.0 <= progress['frac'] <= 1.0 and progress['ref'] == REF
  done = c.wait_for('download', state='done')
  assert done['frac'] == 1.0 and done['bytes'] == 4096
  assert [m['sha256'] for m in c.wait_for('inventory')['models']] == [SHA]

  c.send(id=2, cmd='download', ref=REF)
  assert 'already downloaded' in c.wait_for('reply', id=2)['error']


def test_a_download_can_be_cancelled_and_leaves_nothing_behind(bench):
  bench.registry.pointers[REF] = Pointer(SHA, 2048)
  started = threading.Event()

  def fetch(ref_or_sha, progress, should_stop):
    started.set()
    for _ in range(200):
      if should_stop():
        raise RuntimeError('cancelled')
      progress(0.1)
      time.sleep(0.01)
    raise AssertionError('should_stop was never honoured')

  bench.registry.fetch_impl = fetch
  c = bench.connect()
  c.send(id=1, cmd='download', sha256=SHA)
  c.wait_for('download', state='started')
  assert started.wait(2.0)
  c.send(id=2, cmd='cancel_download', sha256=SHA)
  assert c.wait_for('download', state='cancelled')['sha256'] == SHA
  assert not bench.registry.model_path(SHA).exists()

  c.send(id=3, cmd='cancel_download', sha256=SHA)
  assert 'no download' in c.wait_for('reply', id=3)['error']


def test_a_download_needs_exactly_one_of_ref_and_sha256(bench):
  c = bench.connect()
  c.send(id=1, cmd='download')
  assert 'exactly one' in c.wait_for('reply', id=1)['error']
  c.send(id=2, cmd='download', ref=REF, sha256=SHA)
  assert 'exactly one' in c.wait_for('reply', id=2)['error']


# -- catalog and import ------------------------------------------------------


def test_catalog_refreshes_then_resolves_the_pointers_it_lacks(bench):
  bench.registry.entries = [
    {'name': 'Cinque Terre Model V2', 'short_name': 'CTMV2', 'ref': 'a' * 40,
     'build_time': '2026-09-08T00:00:00Z', 'index': 12},
    {'name': 'BMRLNAP Model v4', 'short_name': 'BMRLNAP', 'ref': REF,
     'build_time': '2026-08-30T00:00:00Z', 'index': 11},
  ]
  bench.registry.resolvable = {'a' * 40: Pointer(OTHER, 10), REF: Pointer(SHA, 20)}
  c = bench.connect()
  c.send(id=1, cmd='catalog', refresh=True)
  assert c.wait_for('reply', id=1)['queued'] is True
  catalog = c.wait_for('catalog')
  assert [m['sha256'] for m in catalog['models']] == [OTHER, SHA]
  assert catalog['error'] is None
  assert bench.registry.refreshes == 1
  assert bench.registry.resolved == [['a' * 40, REF]]


def test_a_failing_catalog_refresh_is_reported_not_lost(bench):
  def boom(refresh=False, max_age=3600.0, opener=None):
    raise OSError('the network is down')

  bench.registry.catalog = boom
  c = bench.connect()
  c.send(id=1, cmd='catalog', refresh=True)
  catalog = c.wait_for('catalog')
  assert 'the network is down' in catalog['error']


def test_a_missing_catalog_does_not_block_the_first_client(bench):
  """The registry fetches when it has nothing cached, whatever max_age says."""
  bench.registry.catalog_path.unlink()

  def slow_and_broken(refresh=False, max_age=3600.0, opener=None):
    time.sleep(1.5)
    raise OSError('the network is down')

  bench.registry.catalog = slow_and_broken
  started = time.monotonic()
  c = bench.connect(drain=False)
  events = [c.read() for _ in range(6)]
  assert time.monotonic() - started < 1.0, 'the on-connect burst waited on the network'
  catalog = events[5]
  assert catalog['event'] == 'catalog' and catalog['models'] == [] and catalog['fetched_at'] is None
  assert catalog['url'].startswith('https://') and len(catalog['default_ref']) == 40
  assert catalog['error'] is None
  # And the fetch it started off the burst reports what happened.
  assert 'the network is down' in c.wait_for('catalog', timeout=6.0)['error']


def test_import_hashes_copies_and_lists(bench, tmp_path):
  source = tmp_path / 'big.onnx'
  source.write_bytes(b'y' * 128)

  def import_model(path, name, progress):
    for frac in (0.0, 0.5, 1.0):
      progress(frac)
    bench.registry.model_path(SHA).write_bytes(path.read_bytes())
    return LocalModel(SHA, 128, name or 'big.onnx', 1757440000.0)

  bench.registry.import_impl = import_model
  c = bench.connect()
  c.send(id=1, cmd='import', path=str(source))
  assert c.wait_for('reply', id=1)['queued'] is True
  assert c.wait_for('import', state='hashing')['sha256'] is None
  assert c.wait_for('import', state='copying')['frac'] >= 0.5
  done = c.wait_for('import', state='done')
  assert done['sha256'] == SHA and done['path'] == str(source)
  assert [m['sha256'] for m in c.wait_for('inventory')['models']] == [SHA]


def test_importing_something_that_is_not_a_file_is_refused(bench, tmp_path):
  c = bench.connect()
  c.send(id=1, cmd='import', path=str(tmp_path / 'nope.onnx'))
  assert 'is not a file' in c.wait_for('reply', id=1)['error']


# -- frames and stats --------------------------------------------------------


def infer_once(session, spec, frame_id: int = 1) -> None:
  payload = P.pack_infer_req(frame_id, 0) + bytes(spec.warped_nbytes + spec.packed_nbytes)
  session.handle(Message(P.Msg.INFER_REQ, frame_id, 0, memoryview(payload)))


def test_a_served_frame_reaches_the_stats_ticker(bench):
  spec = bench.spec
  engine = FakeEngine(spec)
  bench.host.loaded = Loaded(spec.sha256, spec, engine, PolicyQueues(spec),
                             {n: engine.host_input(n) for n in spec.input_shapes})
  session = Session(SimpleNamespace(send=lambda *a, **kw: None), bench.host)
  session.request = Request(spec.sha256, spec.nbytes, spec.frame_skip)
  bench.host.session = session

  c = bench.connect()
  infer_once(session, spec)
  assert engine.calls == 1
  summary = bench.host.frame_stats.summary(60.0, frames_total=session.frames)
  assert summary['frames'] == 1 and summary['gpu_ms']['mean'] == round(1234 / 1e3, 2)
  assert summary['slow'] == 0 and summary['window_s'] == 60.0

  bench.host.emit('link', {'state': 'connected', 'detail': '', 'peer': 'usb'})
  assert c.wait_for('link', state='connected')['peer'] == 'usb'
  infer_once(session, spec, frame_id=2)
  stats = c.wait_for('stats', timeout=4.0)
  assert stats['frames'] >= 1 and stats['fps'] > 0 and stats['window_s'] == 1.0


def test_frame_stats_is_empty_until_a_frame_lands(bench):
  assert bench.host.frame_stats.summary(1.0) is None
  bench.host.frame_stats.record(70_000, 1000)
  summary = bench.host.frame_stats.summary(60.0, frames_total=3)
  assert summary['frames'] == 3 and summary['slow'] == 1
  assert summary['total_ms'] == {'mean': 70.0, 'p99': 70.0, 'max': 70.0}


# -- bad input ---------------------------------------------------------------


@pytest.mark.parametrize('line', [b'not json\n', b'[]\n', b'{"cmd":"status"}\n'])
def test_a_malformed_line_gets_a_reply_with_no_id(bench, line):
  c = bench.connect()
  c.send_raw(line)
  reply = c.wait_for('reply')
  assert reply['id'] is None and reply['ok'] is False and reply['error']


def test_an_unknown_command_is_refused(bench):
  c = bench.connect()
  c.send(id=9, cmd='launch_the_car')
  reply = c.wait_for('reply', id=9)
  assert reply['ok'] is False and 'no command' in reply['error']


def test_every_client_hears_every_event(bench):
  first, second = bench.connect(), bench.connect()
  first.send(id=1, cmd='inventory')
  assert second.wait_for('inventory')['models'] == []


# -- main.py -----------------------------------------------------------------


def test_the_parent_watcher_interrupts_the_main_thread(monkeypatch):
  from jetlink.server import main as M
  interrupts = []
  monkeypatch.setattr(M.time, 'sleep', lambda _: None)
  monkeypatch.setattr(M.os, 'getppid', lambda: 1)      # launchd adopted us: the app is gone
  monkeypatch.setattr(M, '_interrupt_main', lambda: interrupts.append(True))
  M._watch_parent(4242, interval=0.0)
  assert interrupts == [True]


def test_a_missing_gadget_is_logged_once_not_every_poll(caplog, monkeypatch):
  """A box parked offroad polls every 2 s all night; one line is enough."""
  import logging

  pytest.importorskip('usb1')   # the usb opener imports its transport when it is built
  from jetlink.server import main as M
  from jetlink.transport.usbbulk import UsbBulkTransport

  args = SimpleNamespace(vid=0x1209, pid=0x0001, usb_timeout_ms=2000)
  present = [False]
  monkeypatch.setattr(UsbBulkTransport, 'present', lambda vid, pid: present[0])
  monkeypatch.setattr(UsbBulkTransport, 'open', lambda vid, pid, timeout_ms=0: SimpleNamespace())
  opener = M._usb_opener(args)
  with caplog.at_level(logging.DEBUG, logger='jetlink.server'):
    for _ in range(4):
      assert opener() is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ['waiting for a jetlink gadget at 1209:0001']
    assert len([r for r in caplog.records if r.levelno == logging.DEBUG]) == 3

    # A gadget, and then its next absence is worth a line again.
    present[0] = True
    assert opener() is not None
    present[0] = False
    caplog.clear()
    for _ in range(3):
      assert opener() is None
  warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
  assert warnings == ['waiting for a jetlink gadget at 1209:0001']


def test_link_transitions_are_emitted_once_each(tmp_path, monkeypatch):
  from jetlink.server import main as M

  class Stop(Exception):
    pass

  transport = SimpleNamespace(peer='usb', close=lambda: None, drain=lambda t: None,
                              recv=_raise_link_error)
  polls = []

  def opener():
    polls.append(1)
    if len(polls) <= 3:
      return None      # three polls with no gadget is one waiting event
    if len(polls) == 4:
      return transport
    raise Stop

  opener.waiting_detail = 'waiting for a jetlink gadget at 1209:0001'
  host = EngineHost(EngineCache(tmp_path, FakeBackend()))
  seen = []
  host.subscribe(lambda kind, payload: seen.append((kind, payload)) if kind == 'link' else None)
  monkeypatch.setattr(M.time, 'sleep', lambda _: None)
  with pytest.raises(Stop):
    M._serve(EngineCache(tmp_path, FakeBackend()), opener, None, host)
  assert [p['state'] for _, p in seen] == ['waiting', 'connected', 'disconnected']
  assert seen[0][1]['detail'] == 'waiting for a jetlink gadget at 1209:0001'
  assert seen[1][1]['peer'] == 'usb'
  host.close()


def _raise_link_error(**kwargs):
  raise LinkError('the cable came out')
