"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma asks of the registry: the model index, and a model's bytes.

Every test is offline. Network calls go through an injected `opener` that maps
a URL to a fixture.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from jetlink.registry.index import (DEFAULT_BIG_MODEL_REF, INDEX_URL, OTHER_FOLDER, IndexInvalid, NetworkError, Pointer, RegistryError,
                                    VerifyError, fetch_index, parse_index)
from jetlink.registry.lfs import BUCKET, LFS_ENDPOINTS, lfs_download, lfs_hrefs, lfs_resolve

FIXTURES = Path(__file__).parent / 'fixtures'
OID = 'a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525'
SIZE = 765953504
GITLAB = LFS_ENDPOINTS[1]
BLOB = b'onnx' * 1024
BLOB_SHA = hashlib.sha256(BLOB).hexdigest()
HREF = 'https://blob.example/object'


def fixture(name: str) -> bytes:
  return (FIXTURES / name).read_bytes()


class FakeResponse:
  def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
    self._buf = io.BytesIO(data)
    self.status = status
    self.headers = headers or {}

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
    if isinstance(body, FakeResponse):
      return body
    return FakeResponse(body if isinstance(body, bytes) else json.dumps(body).encode())


# --- the model index ---------------------------------------------------------

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40
ONNX_A, ONNX_B = '1' * 64, '2' * 64


def published() -> dict:
  return json.loads(fixture('model_index_v1.json'))


def test_the_published_index_parses():
  index = parse_index(published())
  assert index.models, 'it lists models'
  assert index.default in {m.ref for m in index.models}, 'the index names a listed default'
  assert DEFAULT_BIG_MODEL_REF in {m.ref for m in index.models}, "jetlink's fallback default is listed"
  assert all(m.short_name and m.generation.isdigit() for m in index.models), 'every model has what a catalog needs'
  # every listed model, and every ref, has its ONNX's size
  assert all(index.size(m.oid) == m.size for m in index.models)
  assert all(index.size(p.oid) == p.size for p in index.refs.values())


def test_the_models_keep_their_order_and_fields():
  index = parse_index({'schema': 1, 'models': [
    {'ref': B, 'oid': ONNX_B, 'size': 20, 'name': 'Beta (October 29, 2026)', 'folder': '2026 Deep RL Models', 'short_name': 'B',
     'build_time': '2026-10-29T00:00:00Z'},
    {'ref': A, 'oid': ONNX_A, 'size': 10, 'pr': 39235},
  ]})
  assert [(m.ref, m.name, m.folder, m.short_name, m.build_time, m.pr) for m in index.models] == [
    (B, 'Beta (October 29, 2026)', '2026 Deep RL Models', 'B', '2026-10-29T00:00:00Z', None),
    (A, A[:10], OTHER_FOLDER, '', '', 39235)]


def test_the_default_must_be_listed():
  listed = [{'ref': A, 'oid': ONNX_A, 'size': 1}]
  assert parse_index({'schema': 1, 'default': A, 'models': listed}).default == A
  for default in (B, 'junk', None, 7):
    assert parse_index({'schema': 1, 'default': default, 'models': listed, 'refs': {B: {'oid': ONNX_B, 'size': 2}}}).default is None


@pytest.mark.parametrize('bad', [
  {'oid': ONNX_A, 'size': 1},                     # no ref
  {'ref': 'not-a-ref', 'oid': ONNX_A, 'size': 1},
  {'ref': C, 'oid': 'x' * 64, 'size': 1},
  {'ref': C, 'oid': ONNX_A, 'size': 0},
  {'ref': C, 'oid': ONNX_A, 'size': '1'},
  {'ref': C, 'oid': ONNX_A, 'size': True},
  {'ref': A, 'oid': ONNX_B, 'size': 2},           # a second entry for a ref
  'junk', None, 7,
])
def test_an_entry_it_cannot_use_costs_that_entry(bad):
  index = parse_index({'schema': 1, 'models': [{'ref': A, 'oid': ONNX_A, 'size': 1}, bad]})
  assert [m.ref for m in index.models] == [A]


def test_refs_and_onnx_are_read_and_the_listed_models_join_them():
  index = parse_index({'schema': 1, 'models': [{'ref': A, 'oid': ONNX_A, 'size': 1}],
                       'refs': {B: {'oid': ONNX_B, 'size': 2, 'method': 'weights'}, C: {'oid': 'bad', 'size': 2}, 'x': {}},
                       'onnx': {'3' * 64: {'size': 3, 'export': 'e/1/x.onnx'}, '4' * 64: {'size': -1}}})
  assert index.pointer(B) == Pointer(ONNX_B, 2) and index.pointer(A) == Pointer(ONNX_A, 1)
  assert index.pointer(C) is None
  assert (index.size('3' * 64), index.size(ONNX_A), index.size(ONNX_B), index.size('4' * 64)) == (3, 1, 2, None)


def test_a_ref_is_listed_under_its_onnx():
  index = parse_index({'schema': 1, 'models': [{'ref': A, 'oid': ONNX_A, 'size': 1}],
                       'refs': {B: {'oid': ONNX_A, 'size': 1}, C: {'oid': ONNX_B, 'size': 2}}})
  assert index.listed(A).ref == A and index.listed(B).ref == A
  assert index.listed(C) is None and index.listed('f' * 40) is None


@pytest.mark.parametrize('data, why', [
  ([], 'not a JSON object'),
  ({'models': []}, 'schema None'),
  ({'schema': 2, 'models': [{'ref': A, 'oid': ONNX_A, 'size': 1}]}, 'schema 2'),
  ({'schema': 1, 'models': []}, 'lists no model'),
  ({'schema': 1, 'models': 'junk'}, 'lists no model'),
])
def test_an_index_it_cannot_use_is_refused_whole(data, why):
  # the caller keeps the last one that parsed rather than an empty list
  with pytest.raises(IndexInvalid, match=why):
    parse_index(data)


def test_fetch_index_returns_what_to_keep_and_what_it_says():
  opener = FakeOpener({INDEX_URL: published()})
  data, index = fetch_index(opener=opener)
  assert data == published() and index.models
  assert opener.calls == [INDEX_URL]


@pytest.mark.parametrize('body, error', [(b'<html>', NetworkError), (urllib.error.URLError('offline'), NetworkError),
                                         ({'schema': 9}, IndexInvalid)])
def test_fetch_index_fails_as_a_registry_error(body, error):
  with pytest.raises(error):
    fetch_index(opener=FakeOpener({INDEX_URL: body}))
  assert issubclass(error, RegistryError)


# --- lfs ---------------------------------------------------------------------

def test_comma_s_bucket_is_asked_first():
  assert LFS_ENDPOINTS[0] == BUCKET


def test_lfs_resolve_returns_the_href():
  opener = FakeOpener({f"{GITLAB}/objects/batch": fixture('lfs_batch_response.json')})
  href = lfs_resolve(GITLAB, Pointer(OID, SIZE), opener=opener)
  assert href.startswith('https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/')


def test_lfs_resolve_is_none_when_the_server_lacks_it_or_is_down():
  missing = FakeOpener({f"{GITLAB}/objects/batch": fixture('lfs_batch_missing.json')})
  assert lfs_resolve(GITLAB, Pointer(OID, SIZE), opener=missing) is None
  assert lfs_resolve(GITLAB, Pointer(OID, SIZE), opener=FakeOpener({})) is None


def test_the_bucket_is_read_by_oid_after_its_first_byte():
  url = f"{BUCKET}/resolve/sha256/{OID}"
  assert lfs_resolve(BUCKET, Pointer(OID, SIZE), opener=FakeOpener({url: b'o'})) == url
  assert lfs_resolve(BUCKET, Pointer(OID, SIZE), opener=FakeOpener({url: urllib.error.HTTPError(url, 404, 'nf', {}, None)})) is None


def test_lfs_hrefs_skips_a_server_without_it():
  opener = FakeOpener({f"{GITLAB}/objects/batch": fixture('lfs_batch_response.json')})
  found = list(lfs_hrefs(LFS_ENDPOINTS, Pointer(OID, SIZE), opener=opener))
  assert [endpoint for endpoint, _ in found] == [GITLAB]
  assert opener.calls[0] == f"{BUCKET}/resolve/sha256/{OID}"


def download(tmp_path, blob: bytes = BLOB, oid: str = BLOB_SHA, size: int = len(BLOB), opener=None, **kw) -> Path:
  return lfs_download(HREF, Pointer(oid, size), tmp_path / 'models' / 'model.onnx',
                      opener=opener or FakeOpener({HREF: blob}), **kw)


def test_a_download_takes_its_name_only_once_it_verifies(tmp_path):
  seen = []
  path = download(tmp_path, progress=seen.append)
  assert path == tmp_path / 'models' / 'model.onnx'
  assert path.read_bytes() == BLOB
  assert [p.name for p in path.parent.iterdir()] == ['model.onnx']
  assert seen and seen[-1] == 1.0


def test_a_wrong_hash_leaves_nothing_behind(tmp_path):
  with pytest.raises(VerifyError, match='hash'):
    download(tmp_path, oid='b' * 64)
  assert not list((tmp_path / 'models').iterdir())


def test_a_short_download_leaves_nothing_behind(tmp_path):
  with pytest.raises(VerifyError, match='bytes'):
    download(tmp_path, size=len(BLOB) + 99)
  assert not list((tmp_path / 'models').iterdir())


class RangeOpener:
  """Serves BLOB from the Range a request asks for (206), or whole (200) when
  `honour` is off; `cut` bytes in, the transfer stalls (a read timeout)."""

  def __init__(self, blob: bytes = BLOB, honour: bool = True, cut: int | None = None):
    self.blob, self.honour, self.cut = blob, honour, cut
    self.ranges: list[str | None] = []

  def __call__(self, request, timeout=None, **_):
    rng = dict(request.header_items()).get('Range') if hasattr(request, 'header_items') else None
    self.ranges.append(rng)
    start = int(rng.split('=')[1].rstrip('-')) if rng and self.honour else 0
    body, cut = self.blob[start:], self.cut
    response = FakeResponse(body, 206 if rng and self.honour else 200)
    if cut is not None:
      read = response.read

      def stalling(n=-1):
        nonlocal cut
        if cut <= 0:
          raise TimeoutError('The read operation timed out')
        chunk = read(min(n, cut) if n >= 0 else cut)
        cut -= len(chunk)
        return chunk
      response.read = stalling
    return response


@pytest.fixture
def small_chunks(monkeypatch):
  from jetlink.registry import lfs
  monkeypatch.setattr(lfs, 'CHUNK', 1024)


def part_of(tmp_path) -> Path:
  return tmp_path / 'models' / 'model.onnx.part'


def test_a_stopped_download_keeps_its_part_for_the_next_one(tmp_path, small_chunks):
  # the owner stops a run at ignition; the bytes it got are not thrown away
  stops = iter([False, True])
  with pytest.raises(RegistryError, match='stopped'):
    download(tmp_path, opener=RangeOpener(), should_stop=lambda: next(stops))
  assert part_of(tmp_path).read_bytes() == BLOB[:1024]
  assert not (tmp_path / 'models' / 'model.onnx').exists()


def test_a_stalled_download_carries_on_from_its_part(tmp_path, small_chunks):
  # a stall part way into a 755 MB download must not throw the part away
  with pytest.raises(NetworkError, match='timed out'):
    download(tmp_path, opener=RangeOpener(cut=2048))
  assert part_of(tmp_path).stat().st_size == 2048
  opener = RangeOpener()
  path = download(tmp_path, opener=opener)
  assert opener.ranges == ['bytes=2048-']
  assert path.read_bytes() == BLOB and not part_of(tmp_path).exists()


def test_a_server_that_ignores_the_range_starts_over(tmp_path, small_chunks):
  with pytest.raises(NetworkError):
    download(tmp_path, opener=RangeOpener(cut=1024))
  path = download(tmp_path, opener=RangeOpener(honour=False))
  assert path.read_bytes() == BLOB


def test_a_part_that_was_never_this_model_fails_and_goes(tmp_path):
  part_of(tmp_path).parent.mkdir(parents=True)
  part_of(tmp_path).write_bytes(b'x' * 100)
  with pytest.raises(VerifyError, match='hash'):
    download(tmp_path, opener=RangeOpener())
  assert not part_of(tmp_path).exists()


def test_a_part_longer_than_the_model_is_dropped_and_started_over(tmp_path):
  part_of(tmp_path).parent.mkdir(parents=True)
  part_of(tmp_path).write_bytes(BLOB + b'extra')
  opener = RangeOpener()
  assert download(tmp_path, opener=opener).read_bytes() == BLOB
  assert opener.ranges == [None]


def test_a_whole_part_is_verified_without_asking_again(tmp_path):
  # a run stopped between the last byte and taking the name
  part_of(tmp_path).parent.mkdir(parents=True)
  part_of(tmp_path).write_bytes(BLOB)
  opener = RangeOpener()
  assert download(tmp_path, opener=opener).read_bytes() == BLOB
  assert opener.ranges == []


def test_a_failed_transfer_is_a_network_error(tmp_path):
  with pytest.raises(NetworkError, match='could not download'):
    lfs_download(HREF, Pointer(BLOB_SHA, len(BLOB)), tmp_path / 'model.onnx', opener=FakeOpener({}))
  assert not list(tmp_path.iterdir())
