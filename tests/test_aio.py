"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

jetlink.transport.aio itself: buffer addresses everywhere, and the real
io_submit/io_getevents on Linux, against a pipe standing in for ep2.
"""
import ctypes
import os
import sys

import pytest

from jetlink.transport.aio import IOVECS, Aio, address, layout


def test_an_address_is_where_the_bytes_are_for_read_only_buffers_too():
  data = bytes(range(200))
  for buf, want in ((data, data), (memoryview(data)[17:90], data[17:90]),
                    (bytearray(data), data), (memoryview(bytearray(data))[5:], data[5:])):
    addr, n = address(buf)
    assert n == len(want) and ctypes.string_at(addr, n) == want


def test_an_empty_buffer_has_no_bytes():
  assert address(b'')[1] == 0


def test_a_context_fills_its_structs_where_the_kernel_reads_them(monkeypatch):
  # the real context only builds on Linux, so a mistake in its setup (a view
  # made before its array) would reach a comma with every other test green;
  # this builds it anywhere, with the syscalls faked
  from types import SimpleNamespace

  from jetlink.transport import aio as module
  monkeypatch.setattr(module.platform, 'machine', lambda: 'aarch64')
  monkeypatch.setattr(module.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(syscall=SimpleNamespace(restype=None)))
  done = []

  def call(self, nr, *args):
    if nr == self._submit:
      return args[1].value
    if nr == self._getevents:
      for i, res in enumerate(done):
        self._events[i].res = res
      return len(done)
    return 0

  monkeypatch.setattr(Aio, '_call', call)
  a = Aio(7, 4)
  head, body = bytes(range(20)), bytes(range(20, 100))
  bases = [address(head)[0], address(body)[0]]
  plan = layout((20, 80), 60)   # [head + body[:40]], [body[40:]]
  assert a.submit(plan, bases, 0, 2) == 2
  first, second = a._iocbs[0], a._iocbs[1]
  assert (first.nbytes, first.fildes, first.opcode) == (2, 7, 8)
  assert second.nbytes == 1
  assert [(v.base, v.len) for v in a._iovecs[0:2]] == [(bases[0], 20), (bases[1], 40)]
  assert (a._iovecs[IOVECS].base, a._iovecs[IOVECS].len) == (bases[1] + 40, 40)
  # a window from the second request lands in the first iocb
  assert a.submit(plan, bases, 1, 1) == 1
  assert a._iocbs[0].nbytes == 1 and (a._iovecs[0].base, a._iovecs[0].len) == (bases[1] + 40, 40)
  done.extend([60, -108])
  assert a.reap(2, 0.0) == [60, -108]


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_requests_reach_the_fd_in_order_and_are_reaped():
  r, w = os.pipe()
  aio = Aio(w, 8)
  try:
    payload = os.urandom(3000)
    addr, _ = address(payload)
    assert aio.submit(layout((3000,), 2000), [addr], 0, 2) == 2
    assert sorted(aio.reap(2, 1.0)) == [1000, 2000]
    assert os.read(r, 4000) == payload
    assert aio.reap(0, 0.0) == []
  finally:
    aio.close()
    os.close(r)
    os.close(w)


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_a_bad_fd_fails_the_submit_and_queues_nothing():
  aio = Aio(-1 & 0xFFFFFFFF, 4)
  try:
    addr, _ = address(b'x' * 16)
    with pytest.raises(OSError):
      aio.submit(layout((16,), 16), [addr], 0, 1)
  finally:
    aio.close()


def test_a_layout_cuts_spans_into_requests_of_the_size():
  plan = layout((32, 8, 20000, 4536), 8192)
  assert plan.count == 3
  assert list(plan.pieces) == [3, 1, 2]
  assert [int(plan.length[plan.first[r]:plan.first[r + 1]].sum()) for r in range(3)] == [8192, 8192, 8192]
  assert layout((32, 8, 20000, 4536), 8192) is plan, 'a frame of the same shape reuses it'


def test_a_request_gathers_at_most_iovecs_buffers():
  with pytest.raises(ValueError):
    layout((1,) * (IOVECS + 1), 8192)


@pytest.mark.real_aio
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux AIO')
def test_addon_sized_batch(tmp_path):
  payload = os.urandom(372 * 1024)
  plan = layout((len(payload),), 8192)
  # Every write targets offset zero on a regular file. Check the byte counts
  # here; the FunctionFS fake verifies the order of the byte stream.
  with (tmp_path / 'endpoint').open('wb') as endpoint:
    aio = Aio(endpoint.fileno(), plan.count)
    try:
      assert aio.submit(plan, [address(payload)[0]], 0, plan.count) == plan.count
      assert sum(aio.reap(plan.count, 1)) == len(payload)
    finally:
      aio.close()
