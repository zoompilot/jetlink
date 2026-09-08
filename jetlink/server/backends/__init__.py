"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Which runtime runs the model.

    trt       TensorRT on an NVIDIA GPU: the Jetson, and a desktop or laptop
    ort       onnxruntime: CoreML on Apple silicon, CUDA or CPU elsewhere
    tinygrad  tinygrad on whatever it drives: Metal, CUDA, AMD

Runtimes are imported only when chosen, never here: the tests import the
server on machines with none of them, and a Jetson must not pay for a Mac's
imports.

`auto` prefers TensorRT, then tinygrad, then onnxruntime. On a Mac the choice
between the last two is measured (docs/platforms.md): CoreML through
onnxruntime runs the frame in 39 ms to tinygrad's 65 on an M1 Pro, but takes
ten minutes to create its session every time the process starts, against a
second for tinygrad's pickle. A server that restarts has to be usable again
before the car is out of the driveway, so tinygrad is the default and CoreML
is the opt-in for a long-lived server that wants the frame time.
"""
from __future__ import annotations

import importlib.util
import logging

from jetlink.server.backends.base import Backend

log = logging.getLogger('jetlink.backends')

NAMES = ('trt', 'tinygrad', 'ort')


def _importable(module: str) -> bool:
  try:
    return importlib.util.find_spec(module) is not None
  except (ImportError, ValueError):
    return False


def _make(name: str, device: str) -> Backend:
  if name == 'trt':
    from jetlink.server.backends.trt import TrtBackend
    return TrtBackend(device)
  if name == 'ort':
    from jetlink.server.backends.ort import OrtBackend
    return OrtBackend(device)
  if name == 'tinygrad':
    from jetlink.server.backends.tinygrad import TinygradBackend
    return TinygradBackend(device)
  raise ValueError(f"unknown backend {name!r}; one of {NAMES} or auto")


def available() -> list[str]:
  """Backends whose runtime is installed, in auto's order of preference."""
  found = []
  if _importable('tensorrt') and (_importable('cuda.bindings') or _importable('cuda')):
    found.append('trt')
  if _importable('tinygrad'):
    found.append('tinygrad')
  if _importable('onnxruntime'):
    found.append('ort')
  return found


def select(name: str = 'auto', device: str = 'auto') -> Backend:
  """The backend to serve with. A named backend that will not come up raises;
  `auto` moves on to the next and says why."""
  if name != 'auto':
    return _make(name, device)
  candidates = available()
  if not candidates:
    raise RuntimeError("no inference runtime is installed: pip install one of "
                       "'jetlink[trt]', 'jetlink[ort]' or 'jetlink[tinygrad]'")
  reasons = []
  for candidate in candidates:
    try:
      backend = _make(candidate, device)
    except Exception as e:
      reasons.append(f"{candidate}: {type(e).__name__}: {e}")
      log.warning("backend %s not used: %s", candidate, e)
      continue
    log.info("backend %s %s on %s", backend.name, backend.describe()['runtime_version'],
             backend.describe()['device'])
    return backend
  raise RuntimeError("no backend came up:\n  " + "\n  ".join(reasons))
