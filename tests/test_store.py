"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

queues.store() casts the frame through a lookup on numpy 1.x and with numpy's
own cast otherwise; both must give the same bits.
"""
from __future__ import annotations

import numpy as np

from jetlink import queues


def test_lookup_and_cast_store_the_same_bits(monkeypatch):
  src = np.arange(256, dtype=np.uint8).repeat(3).reshape(2, 6, 64)
  out = {}
  for lookup in (True, False):
    monkeypatch.setattr(queues, '_LOOKUP', lookup)
    dest = np.empty(src.shape, np.float16)
    queues.store(dest, src)
    out[lookup] = dest.view(np.uint16).copy()
  assert np.array_equal(out[True], out[False])
  assert np.array_equal(out[False].view(np.float16), src.astype(np.float16))
