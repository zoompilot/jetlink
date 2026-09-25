"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The model registry: the catalog, lfs pointers, downloads, and what is on disk.

Every test is offline. Network calls go through an injected `opener` that maps
a URL to a fixture, and the one code path that cannot take an opener (the CLI)
gets `urllib.request.urlopen` patched.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from jetlink.registry import (COMMIT_PATCH_URL, DRIVING_MODELS_TREE_URL, LFS_ENDPOINTS, POINTER_URL, Pointer, Registry, RegistryError, VerifyError,
                              fetch_pointer, lfs_resolve, parse_catalog, parse_pointer_text)
from jetlink.registry.catalog import CATALOG_URL, NetworkError
from jetlink.registry.cli import main as cli_main
from jetlink.server.cache import EngineCache
from jetlink.spec import ModelSpec
from tests.fake_backend import FakeBackend

FIXTURES = Path(__file__).parent / 'fixtures'
REF = 'f877d7a0ccc3cce943c76e285214c020cd65c899'
OID = 'a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525'
SIZE = 765953504
NEWEST = '37bfa1413edcdc2e8844984b83727c33f81d8f46'
BLOB = b'onnx' * 1024
BLOB_SHA = hashlib.sha256(BLOB).hexdigest()
SMALL_REF = 'a' * 40


def fixture(name: str) -> bytes:
  return (FIXTURES / name).read_bytes()


class FakeResponse:
  def __init__(self, data: bytes, status: int = 200):
    self._buf = io.BytesIO(data)
    self.status = status

  def read(self, n: int = -1) -> bytes:
    return self._buf.read(n if n is not None and n >= 0 else -1)

  def __enter__(self):
    return self

  def __exit__(self, *_):
    return False


class FakeOpener:
  """A urlopen that serves fixtures and refuses everything else."""

  def __init__(self, routes: dict):
    self.routes = routes
    self.calls: list[str] = []

  def __call__(self, url, timeout=None, **_):
    url = url.full_url if hasattr(url, 'full_url') else url
    self.calls.append(url)
    body = self.routes.get(url)
    if body is None:
      raise urllib.error.URLError(f"no route for {url}")
    if isinstance(body, Exception):
      raise body
    if callable(body):
      body = body()
    return FakeResponse(body if isinstance(body, bytes) else json.dumps(body).encode())


def catalog_opener(**extra) -> FakeOpener:
  return FakeOpener({CATALOG_URL: fixture('catalog_chestnut_v25.json'), **extra})


def batch(oid: str, size: int, href: str | None) -> bytes:
  """One of the batch fixtures, retargeted at another object."""
  name = 'lfs_batch_missing.json' if href is None else 'lfs_batch_response.json'
  payload = json.loads(fixture(name))
  payload['objects'][0]['oid'] = oid
  payload['objects'][0]['size'] = size
  if href is not None:
    payload['objects'][0]['actions']['download']['href'] = href
  return json.dumps(payload).encode()


def small_pointer_text(oid: str = BLOB_SHA, size: int = len(BLOB)) -> bytes:
  return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()


def small_routes(oid: str = BLOB_SHA, size: int = len(BLOB), blob: bytes = BLOB) -> dict:
  """The first endpoint has nothing, the second serves the object."""
  return {
    POINTER_URL.format(ref=SMALL_REF): small_pointer_text(oid, size),
    f"{LFS_ENDPOINTS[0]}/objects/batch": batch(oid, size, None),
    f"{LFS_ENDPOINTS[1]}/objects/batch": batch(oid, size, 'https://blob.example/object'),
    'https://blob.example/object': blob,
  }


def make_spec(sha256: str) -> ModelSpec:
  return ModelSpec(sha256=sha256, nbytes=1234, frame_skip=4,
                   input_shapes={'img': (1, 12, 128, 256)},
                   output_shapes={'outputs': (1, 18452)},
                   output_slices={'hidden_state': slice(2066, 18450)}, checkpoint='b9facbcc')


# --- catalog -----------------------------------------------------------------

def test_parse_catalog_keeps_the_big_models_newest_first():
  models = parse_catalog(json.loads(fixture('catalog_chestnut_v25.json')))
  assert len(models) == 13
  assert models[0].ref == NEWEST
  assert models[0].short_name == 'CTMV2'
  assert [m.index for m in models] == sorted((m.index for m in models), reverse=True)


@pytest.mark.parametrize('mutate', [
  {'minimum_selector_version': '18'},
  {'ref': 'not-a-commit'},
  {'is_big': False},
])
def test_parse_catalog_drops_a_bundle_it_cannot_use(mutate):
  data = json.loads(fixture('catalog_chestnut_v25.json'))
  data['bundles'][0].update(mutate)
  models = parse_catalog(data)
  assert len(models) == 12
  assert all(m.ref != 'fa0c6876d3cf070e91e25e5353ceadc68a5b3285' for m in models)


def test_parse_catalog_survives_rubbish():
  assert parse_catalog({}) == []
  assert parse_catalog({'bundles': 'nope'}) == []
  assert parse_catalog({'bundles': [None, 5, {'ref': REF, 'minimum_selector_version': 'x', 'is_big': True}]}) == []


def test_a_duplicate_ref_is_listed_once():
  data = json.loads(fixture('catalog_chestnut_v25.json'))
  data['bundles'].append(dict(data['bundles'][0], display_name='A copy', index=99))
  models = parse_catalog(data)
  assert len(models) == 13
  assert all(m.name != 'A copy' for m in models)


# --- pointers ----------------------------------------------------------------

def test_parse_pointer_text_reads_the_fixture():
  pointer = parse_pointer_text(fixture('pointer_f877d7a0.txt').decode())
  assert pointer == Pointer(OID, SIZE)


@pytest.mark.parametrize('text', [
  'not a pointer at all',
  f"oid sha256:{OID}\n",                       # no size
  'oid sha256:nothex\nsize 12\n',
  f"oid sha256:{OID}\nsize twelve\n",
  'x' * 5000,                                  # an onnx served where a pointer was expected
])
def test_parse_pointer_text_refuses_anything_else(text):
  assert parse_pointer_text(text) is None


def test_resolve_fetches_a_pointer_once_and_keeps_it(tmp_path):
  opener = FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')})
  registry = Registry(tmp_path)
  assert registry.resolve(REF, opener=opener) == Pointer(OID, SIZE)
  assert Registry(tmp_path).resolve(REF, opener=opener) == Pointer(OID, SIZE)
  assert len(opener.calls) == 1


def test_resolve_missing_reports_the_failures_and_keeps_the_rest(tmp_path):
  good, bad = NEWEST, REF
  opener = FakeOpener({POINTER_URL.format(ref=good): fixture('pointer_f877d7a0.txt')})
  registry = Registry(tmp_path)

  out = registry.resolve_missing([good, bad], opener=opener)

  assert out[good] == Pointer(OID, SIZE)
  assert isinstance(out[bad], Exception)
  assert json.loads((tmp_path / 'registry' / 'pointers.json').read_text()) == {good: {'oid': OID, 'size': SIZE}}


# --- the catalog payload -----------------------------------------------------

def test_catalog_is_cached_until_it_goes_stale(tmp_path):
  opener = catalog_opener()
  registry = Registry(tmp_path)

  payload = registry.catalog(opener=opener)
  assert len(payload['models']) == 13
  assert payload['error'] is None and payload['fetched_at'] is not None
  assert payload['default_ref'] == 'f877d7a0ccc3cce943c76e285214c020cd65c899'

  registry.catalog(opener=opener)
  assert opener.calls.count(CATALOG_URL) == 1, 'a fresh cache must not go to the network'

  registry.catalog(max_age=-1, opener=opener)
  assert opener.calls.count(CATALOG_URL) == 2


def _bundle(ref: str, index: int, selector: str = '19', name: str = '', **extra) -> dict:
  return {'ref': ref, 'index': index, 'minimum_selector_version': selector, 'is_big': True,
          'display_name': name or ref[:6], 'short_name': name[:4], 'generation': '12', 'environment': 'development',
          'runner': 'tinygrad', 'build_time': '2026-09-25T00:00:00Z', 'overrides': {'folder': 'Master Models'},
          'models': [{'type': 'chunked', 'artifact': {'file_name': f'{ref[:6]}.pkl'}}], **extra}


class TestNewerCatalogs:
  """A model sunnypilot publishes after this release is still listed."""

  def url(self, v):
    from jetlink.registry.catalog import CATALOG_URL_TEMPLATE
    return CATALOG_URL_TEMPLATE.format(version=v)

  def test_versions_are_probed_up_to_the_first_missing_one(self):
    from jetlink.registry.catalog import CATALOG_VERSION, newer_catalogs
    v = CATALOG_VERSION
    opener = FakeOpener({self.url(v + 1): {'bundles': []}, self.url(v + 2): {'bundles': []},
                         self.url(v + 3): urllib.error.HTTPError(self.url(v + 3), 404, 'Not Found', {}, None)})
    assert len(newer_catalogs(opener=opener)) == 2
    assert opener.calls == [self.url(v + 1), self.url(v + 2), self.url(v + 3)]

  def test_an_outage_past_the_pin_keeps_what_was_found(self):
    from jetlink.registry.catalog import CATALOG_VERSION, newer_catalogs
    opener = FakeOpener({self.url(CATALOG_VERSION + 1): {'bundles': []}})
    assert len(newer_catalogs(opener=opener)) == 1

  def test_the_merge_keeps_builds_at_our_version_and_adds_the_rest_for_an_accelerator(self):
    from jetlink.registry.catalog import ACCELERATOR_ONLY, merge_catalogs
    a, b, c = 'a' * 40, 'b' * 40, 'c' * 40
    pinned = {'tinygrad_ref': 'pinned', 'bundles': [_bundle(a, 1), _bundle(b, 2)]}
    next_runtime = {'tinygrad_ref': 'next', 'bundles': [_bundle(a, 1, '20'), _bundle(b, 2, '20'), _bundle(c, 3, '20', 'Old name')]}
    newest = {'tinygrad_ref': 'newer', 'bundles': [_bundle(c, 3, '20', 'Cinque Terre V4')]}
    merged = merge_catalogs([pinned, next_runtime, newest])
    assert merged['tinygrad_ref'] == 'pinned'
    by_ref = {x['ref']: x for x in merged['bundles']}
    assert by_ref[a] == pinned['bundles'][0] and by_ref[b] == pinned['bundles'][1]
    assert by_ref[c]['display_name'] == 'Cinque Terre V4'
    assert by_ref[c]['minimum_selector_version'] == '19' and by_ref[c]['models'] == []
    assert by_ref[c]['overrides'] == {'folder': 'Master Models', ACCELERATOR_ONLY: '1'}
    assert [m.ref for m in parse_catalog(merged)] == [c, b, a]

  def test_the_registry_lists_a_model_only_a_newer_catalog_has(self, tmp_path):
    from jetlink.registry.catalog import CATALOG_VERSION
    fresh = 'e' * 40
    opener = catalog_opener(**{self.url(CATALOG_VERSION + 1): {'bundles': [_bundle(fresh, 99, '20', 'Cinque Terre V4')]}})
    models = Registry(tmp_path).catalog(opener=opener)['models']
    assert models[0]['ref'] == fresh and models[0]['name'] == 'Cinque Terre V4'
    assert len(models) == 14


def test_a_failed_refresh_keeps_the_previous_list(tmp_path):
  registry = Registry(tmp_path)
  registry.catalog(opener=catalog_opener())

  payload = registry.catalog(refresh=True, opener=FakeOpener({}))

  assert payload['error'] is not None and CATALOG_URL in payload['error']
  assert len(payload['models']) == 13


def test_an_empty_cache_and_no_network_is_an_empty_list(tmp_path):
  payload = Registry(tmp_path).catalog(opener=FakeOpener({}))
  assert payload['models'] == [] and payload['fetched_at'] is None and payload['error']


def test_the_catalog_carries_a_resolved_pointer(tmp_path):
  registry = Registry(tmp_path)
  registry.resolve(REF, opener=FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')}))
  payload = registry.catalog(opener=catalog_opener())
  entry = next(m for m in payload['models'] if m['ref'] == REF)
  assert entry['sha256'] == OID and entry['bytes'] == SIZE
  assert next(m for m in payload['models'] if m['ref'] == NEWEST)['sha256'] is None


def test_name_for_finds_the_catalog_name(tmp_path):
  registry = Registry(tmp_path)
  registry.catalog(opener=catalog_opener())
  registry.resolve(REF, opener=FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')}))
  assert registry.name_for(OID) == ('BMRLNAP Model v4 (August 30, 2026)', REF)
  assert registry.name_for('b' * 64) == (None, None)


# --- lfs ---------------------------------------------------------------------

def test_lfs_resolve_returns_the_href():
  opener = FakeOpener({f"{LFS_ENDPOINTS[0]}/objects/batch": fixture('lfs_batch_response.json')})
  href = lfs_resolve(LFS_ENDPOINTS[0], Pointer(OID, SIZE), opener=opener)
  assert href.startswith('https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/')


def test_lfs_resolve_is_none_when_the_server_lacks_it_or_is_down():
  missing = FakeOpener({f"{LFS_ENDPOINTS[0]}/objects/batch": fixture('lfs_batch_missing.json')})
  assert lfs_resolve(LFS_ENDPOINTS[0], Pointer(OID, SIZE), opener=missing) is None
  assert lfs_resolve(LFS_ENDPOINTS[0], Pointer(OID, SIZE), opener=FakeOpener({})) is None


# --- fetch -------------------------------------------------------------------

def test_fetch_falls_through_to_the_server_that_has_it(tmp_path):
  registry = Registry(tmp_path)
  opener = FakeOpener(small_routes())
  seen = []

  path = registry.fetch(SMALL_REF, progress=seen.append, opener=opener)

  assert path == registry.model_path(BLOB_SHA)
  assert path.read_bytes() == BLOB
  assert not list(tmp_path.glob('models/*.part'))
  assert seen and seen[-1] == 1.0
  assert f"{LFS_ENDPOINTS[0]}/objects/batch" in opener.calls


def test_fetch_of_a_model_already_on_disk_touches_no_network(tmp_path):
  registry = Registry(tmp_path)
  registry.fetch(SMALL_REF, opener=FakeOpener(small_routes()))
  offline = FakeOpener({POINTER_URL.format(ref=SMALL_REF): small_pointer_text()})
  assert registry.fetch(SMALL_REF, opener=offline).read_bytes() == BLOB


def test_a_wrong_hash_leaves_nothing_behind(tmp_path):
  registry = Registry(tmp_path)
  wrong = 'b' * 64
  with pytest.raises(VerifyError, match='hash'):
    registry.fetch(SMALL_REF, opener=FakeOpener(small_routes(oid=wrong)))
  assert not list((tmp_path / 'models').iterdir())


def test_a_short_download_leaves_nothing_behind(tmp_path):
  registry = Registry(tmp_path)
  with pytest.raises(VerifyError, match='bytes'):
    registry.fetch(SMALL_REF, opener=FakeOpener(small_routes(size=len(BLOB) + 99)))
  assert not list((tmp_path / 'models').iterdir())


def test_a_cancelled_download_leaves_nothing_behind(tmp_path):
  registry = Registry(tmp_path)
  stops = iter([False, True, True])
  with pytest.raises(RegistryError, match='cancelled'):
    registry.fetch(SMALL_REF, should_stop=lambda: next(stops), opener=FakeOpener(small_routes()))
  assert not list((tmp_path / 'models').iterdir())


def test_no_server_has_it(tmp_path):
  routes = small_routes()
  routes[f"{LFS_ENDPOINTS[1]}/objects/batch"] = batch(BLOB_SHA, len(BLOB), None)
  with pytest.raises(NetworkError, match='no LFS server'):
    Registry(tmp_path).fetch(SMALL_REF, opener=FakeOpener(routes))


def test_fetching_by_sha256_needs_a_known_size(tmp_path):
  registry = Registry(tmp_path)
  with pytest.raises(RegistryError, match='fetch by catalog ref'):
    registry.fetch(BLOB_SHA, opener=FakeOpener({}))
  with pytest.raises(RegistryError, match='neither'):
    registry.fetch('nothex', opener=FakeOpener({}))


# --- a commit that ships a precompiled pkl -----------------------------------
# Cinque Terre V3, as github and huggingface served it on 2026-09-25.

V3_REF = 'bf3e3631b3f91d92a1020a5e0dd4298b93ff4244'
V3_OID = '404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4'
V3_SIZE = 766354845
V3_FOLDER = 'f78ed37d-afad-4dbc-8050-40ea885eedde'


def not_found(url):
  return urllib.error.HTTPError(url, 404, 'Not Found', {}, None)


def patch_head(subject: str) -> bytes:
  return (f"From {V3_REF} Mon Sep 17 00:00:00 2001\nFrom: Bruce Wayne <x@example.com>\n"
          f"Date: Tue, 15 Sep 2026 23:30:35 -0700\nSubject: [PATCH] {subject}\n\n---\n"
          " openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl | 2 +-\n").encode()


def onnx_entry(path: str, oid: str = V3_OID, size: int = V3_SIZE) -> dict:
  return {'type': 'file', 'path': f"{path}/big_driving_supercombo.onnx", 'size': size,
          'lfs': {'oid': oid, 'size': size, 'pointerSize': 134}}


def export_routes(subject='Use f78ed37d for the precompiled eGPU driving model', folder_files=None) -> dict:
  url = POINTER_URL.format(ref=V3_REF)
  return {
    url: not_found(url),
    COMMIT_PATCH_URL.format(ref=V3_REF): patch_head(subject),
    DRIVING_MODELS_TREE_URL: [{'type': 'directory', 'path': '1a421175-db71-4e3d-9d62-e2166421b02b'},
                              {'type': 'directory', 'path': V3_FOLDER},
                              {'type': 'file', 'path': 'README.md', 'size': 21}],
    f"{DRIVING_MODELS_TREE_URL}/{V3_FOLDER}?recursive=true": folder_files or [
      {'type': 'directory', 'path': f"{V3_FOLDER}/12864"}, onnx_entry(f"{V3_FOLDER}/12864")],
  }


def test_a_commit_without_the_onnx_resolves_to_the_export_its_subject_names(tmp_path):
  registry = Registry(tmp_path)
  assert registry.resolve(V3_REF, opener=FakeOpener(export_routes())) == Pointer(V3_OID, V3_SIZE)
  # kept for good like any other pointer
  assert Registry(tmp_path).resolve(V3_REF, opener=FakeOpener({})) == Pointer(V3_OID, V3_SIZE)


def test_the_export_repo_is_the_last_lfs_server_asked():
  assert LFS_ENDPOINTS[-1] == 'https://huggingface.co/commaai/openpilot_driving_models.git/info/lfs'


def test_a_subject_that_names_the_checkpoint_picks_among_several(tmp_path):
  files = [onnx_entry(f"{V3_FOLDER}/12000", oid='1' * 64), onnx_entry(f"{V3_FOLDER}/12864")]
  opener = FakeOpener(export_routes(subject=f"{V3_FOLDER}/12864", folder_files=files))
  assert fetch_pointer(V3_REF, opener=opener) == Pointer(V3_OID, V3_SIZE)
  opener = FakeOpener(export_routes(folder_files=files))
  with pytest.raises(RegistryError, match='2 copies'):
    fetch_pointer(V3_REF, opener=opener)


def test_a_folded_subject_is_read_whole_and_the_diffstat_is_not():
  from jetlink.registry.lfs import commit_subject
  head = patch_head('Use f78ed37d for the precompiled eGPU\n driving model')
  assert commit_subject(V3_REF, opener=FakeOpener({COMMIT_PATCH_URL.format(ref=V3_REF): head})) == \
    'Use f78ed37d for the precompiled eGPU driving model'


@pytest.mark.parametrize(('subject', 'match'), [
  ('Update tinygrad and use retargetable model artifacts (#38933)', 'names no export'),
  ('Use 0badc0de for the precompiled eGPU driving model', 'no folder'),
])
def test_a_subject_that_leads_nowhere_says_so(subject, match):
  with pytest.raises(RegistryError, match=match):
    fetch_pointer(V3_REF, opener=FakeOpener(export_routes(subject=subject)))


def test_only_a_missing_file_falls_back_and_an_outage_does_not():
  url = POINTER_URL.format(ref=V3_REF)
  routes = export_routes()
  routes[url] = urllib.error.HTTPError(url, 503, 'Unavailable', {}, None)
  opener = FakeOpener(routes)
  with pytest.raises(NetworkError):
    fetch_pointer(V3_REF, opener=opener)
  assert opener.calls == [url]


# --- importing ---------------------------------------------------------------

def test_import_hashes_copies_and_records(tmp_path):
  registry = Registry(tmp_path)
  source = tmp_path / 'big_driving_supercombo.onnx'
  source.write_bytes(BLOB)
  seen = []

  local = registry.import_model(source, progress=seen.append)

  assert local.sha256 == BLOB_SHA and local.bytes == len(BLOB)
  assert local.name == 'big_driving_supercombo'
  assert registry.model_path(BLOB_SHA).read_bytes() == BLOB
  assert seen[-1] == 1.0

  registry.import_model(source, name='again')
  assert [m.name for m in registry.local_models()] == ['again']
  assert registry.name_for(BLOB_SHA) == ('again', None)


def test_import_refuses_a_file_that_is_not_an_onnx(tmp_path):
  source = tmp_path / 'model.bin'
  source.write_bytes(BLOB)
  with pytest.raises(RegistryError, match='not an .onnx'):
    Registry(tmp_path).import_model(source)


# --- inventory ---------------------------------------------------------------

def build_cache(root: Path) -> tuple[str, str]:
  """A cache with one fake artifact, one ort artifact, a model and a .part."""
  fake_sha, ort_sha = 'b' * 64, 'c' * 64
  cache = EngineCache(root, FakeBackend())
  entry = cache.entry(fake_sha)
  entry.path.write_bytes(b'plan')
  entry.write_meta({'backend': 'fake', 'device': 'test', 'built_at': '2026-09-08T21:19:15Z',
                    'build_seconds': 548.9, 'spec': make_spec(fake_sha).to_dict()})

  artifact = cache.engines / f"{ort_sha[:16]}.ort1.29.0.coreml-Apple_M1_Pro.ortcache"
  (artifact / 'inner').mkdir(parents=True)
  (artifact / 'inner' / 'model').write_bytes(b'x' * 100)
  (artifact / 'session').write_bytes(b'y' * 50)
  artifact.with_suffix('.json').write_text(json.dumps({
    'backend': 'ort', 'onnxruntime': '1.29.0', 'device': 'coreml-Apple_M1_Pro',
    'built_at': '2026-09-08T21:19:15Z', 'build_seconds': 548.9, 'spec': make_spec(ort_sha).to_dict()}))

  (cache.engines / 'nospec.fake').write_bytes(b'x')
  (cache.engines / 'nospec.json').write_text('{"backend": "fake"}')
  cache.model_path(fake_sha).write_bytes(b'onnx bytes')
  (cache.models / f"{ort_sha[:16]}.onnx.part").write_bytes(b'half')
  return fake_sha, ort_sha


def test_inventory_lists_both_backends_and_marks_the_current_one(tmp_path):
  fake_sha, ort_sha = build_cache(tmp_path)
  registry = Registry(tmp_path)

  payload = registry.inventory(EngineCache(tmp_path, FakeBackend()))

  keys = {a['sha256']: a for a in payload['artifacts']}
  assert set(keys) == {fake_sha, ort_sha}
  assert keys[fake_sha]['current'] is True and keys[ort_sha]['current'] is False
  assert keys[ort_sha]['bytes'] == 150, 'a directory artifact counts everything under it'
  assert keys[ort_sha]['runtime_version'] == '1.29.0'
  assert keys[ort_sha]['checkpoint'] == 'b9facbcc'
  assert keys[ort_sha]['key'] == f"{ort_sha[:16]}.ort1.29.0.coreml-Apple_M1_Pro"

  assert [m['sha256'] for m in payload['models']] == [fake_sha], 'a .part is not a model'
  assert payload['disk']['engines_bytes'] == 154
  assert payload['disk']['models_bytes'] == len(b'onnx bytes')
  assert payload['loaded'] is None and payload['last_loaded'] is None


def test_inventory_without_a_backend_marks_nothing_current(tmp_path):
  build_cache(tmp_path)
  payload = Registry(tmp_path).inventory()
  assert all(a['current'] is False for a in payload['artifacts'])


def test_a_model_of_unknown_identity_is_listed_by_its_prefix(tmp_path):
  registry = Registry(tmp_path)
  (tmp_path / 'models' / f"{'d' * 16}.onnx").write_bytes(b'x')
  (tmp_path / 'models' / 'not-a-model.onnx').write_bytes(b'x')
  models = registry.inventory()['models']
  assert [m['sha256'] for m in models] == ['d' * 16]
  assert models[0]['name'] is None and models[0]['ref'] is None


def test_last_loaded_comes_from_the_marker(tmp_path):
  fake_sha, _ = build_cache(tmp_path)
  EngineCache(tmp_path, FakeBackend()).remember_loaded(fake_sha, 4)
  assert Registry(tmp_path).inventory()['last_loaded'] == fake_sha


# --- removal -----------------------------------------------------------------

def test_remove_takes_every_backend_the_model_and_the_marker(tmp_path):
  fake_sha, _ = build_cache(tmp_path)
  cache = EngineCache(tmp_path, FakeBackend())
  other = cache.engines / f"{fake_sha[:16]}.trt10.3.orin.plan"
  other.write_bytes(b'plan')
  other.with_suffix('.json').write_text('{}')
  cache.remember_loaded(fake_sha, 4)
  registry = Registry(tmp_path)

  registry.remove(fake_sha, artifacts=True, model=True)

  assert not list(cache.engines.glob(f"{fake_sha[:16]}.*"))
  assert not cache.model_path(fake_sha).exists()
  assert not (tmp_path / 'last-loaded.json').exists()
  registry.remove(fake_sha, artifacts=True, model=True)   # nothing left to remove is not an error


def test_remove_of_the_model_only_keeps_the_engines(tmp_path):
  fake_sha, _ = build_cache(tmp_path)
  registry = Registry(tmp_path)
  registry.remove(fake_sha, artifacts=False, model=True)
  assert not registry.model_path(fake_sha).exists()
  assert [a['sha256'] for a in registry.inventory()['artifacts'] if a['sha256'] == fake_sha]


def test_removing_an_imported_model_drops_its_local_record(tmp_path):
  registry = Registry(tmp_path)
  source = tmp_path / 'mine.onnx'
  source.write_bytes(BLOB)
  registry.import_model(source, name='mine')
  keeper = tmp_path / 'other.onnx'
  keeper.write_bytes(b'other bytes')
  other = registry.import_model(keeper, name='other')

  registry.remove(BLOB_SHA, artifacts=False, model=True)

  assert [m.sha256 for m in registry.local_models()] == [other.sha256]
  assert registry.name_for(BLOB_SHA) == (None, None)
  assert [m['sha256'] for m in registry.inventory()['models']] == [other.sha256]


def test_removing_the_artifacts_only_keeps_the_local_record(tmp_path):
  registry = Registry(tmp_path)
  source = tmp_path / 'mine.onnx'
  source.write_bytes(BLOB)
  registry.import_model(source, name='mine')
  registry.remove(BLOB_SHA, artifacts=True, model=False)
  assert [m.name for m in registry.local_models()] == ['mine']


def test_remove_of_a_directory_artifact(tmp_path):
  _, ort_sha = build_cache(tmp_path)
  registry = Registry(tmp_path)
  registry.remove(ort_sha, artifacts=True, model=False)
  assert not list((tmp_path / 'engines').glob(f"{ort_sha[:16]}.*"))


# --- the cli -----------------------------------------------------------------

def test_cli_list_json(tmp_path, capsys, monkeypatch):
  monkeypatch.setattr(urllib.request, 'urlopen', catalog_opener())
  assert cli_main(['list', '--json', '--cache', str(tmp_path)]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert len(payload['models']) == 13 and payload['models'][0]['ref'] == NEWEST


def test_cli_list_table_shows_what_is_downloaded(tmp_path, capsys, monkeypatch):
  registry = Registry(tmp_path)
  registry.catalog(opener=catalog_opener())
  registry.resolve(REF, opener=FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')}))
  registry.model_path(OID).write_bytes(b'x')

  monkeypatch.setattr(urllib.request, 'urlopen', FakeOpener({}))
  assert cli_main(['list', '--cache', str(tmp_path)]) == 0

  line = next(li for li in capsys.readouterr().out.splitlines() if REF[:10] in li)
  assert line.endswith('downloaded') and '730 MB' in line


def test_cli_resolve(tmp_path, capsys, monkeypatch):
  monkeypatch.setattr(urllib.request, 'urlopen', FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')}))
  assert cli_main(['resolve', REF, '--json', '--cache', str(tmp_path)]) == 0
  assert json.loads(capsys.readouterr().out) == {'ref': REF, 'sha256': OID, 'bytes': SIZE}
  assert cli_main(['resolve', 'nothex', '--cache', str(tmp_path)]) == 1


def test_cli_rm_needs_to_be_told_what_to_remove(tmp_path):
  assert cli_main(['rm', 'b' * 64, '--cache', str(tmp_path)]) == 1
  assert cli_main(['rm', 'nothex', '--model', '--cache', str(tmp_path)]) == 1


def test_cli_fetch_and_inventory(tmp_path, capsys, monkeypatch):
  monkeypatch.setattr(urllib.request, 'urlopen', FakeOpener(small_routes()))
  assert cli_main(['fetch', SMALL_REF, '--cache', str(tmp_path)]) == 0
  assert Path(capsys.readouterr().out.strip()).read_bytes() == BLOB

  assert cli_main(['inventory', '--json', '--cache', str(tmp_path)]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert [m['sha256'] for m in payload['models']] == [BLOB_SHA]


def test_cli_import(tmp_path, capsys):
  source = tmp_path / 'mine.onnx'
  source.write_bytes(BLOB)
  assert cli_main(['import', str(source), '--name', 'mine', '--cache', str(tmp_path / 'cache')]) == 0
  sha, path = capsys.readouterr().out.split()
  assert sha == BLOB_SHA and Path(path).read_bytes() == BLOB


def test_cli_exit_codes(tmp_path, monkeypatch):
  monkeypatch.setattr(urllib.request, 'urlopen', FakeOpener({}))
  assert cli_main(['fetch', SMALL_REF, '--cache', str(tmp_path)]) == 2, 'a network failure is 2'
  monkeypatch.setattr(urllib.request, 'urlopen', FakeOpener(small_routes(oid='b' * 64)))
  assert cli_main(['fetch', SMALL_REF, '--cache', str(tmp_path)]) == 3, 'bytes that do not verify are 3'


def test_state_files_are_written_atomically(tmp_path):
  registry = Registry(tmp_path)
  registry.catalog(opener=catalog_opener())
  registry.resolve(REF, opener=FakeOpener({POINTER_URL.format(ref=REF): fixture('pointer_f877d7a0.txt')}))
  assert not list((tmp_path / 'registry').glob('*.tmp'))
  cached = json.loads((tmp_path / 'registry' / 'catalog.json').read_text())
  assert cached['url'] == CATALOG_URL and time.time() - cached['fetched_at'] < 60
