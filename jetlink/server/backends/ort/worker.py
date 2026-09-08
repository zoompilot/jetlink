"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The process that holds an onnxruntime session.

onnxruntime keeps the GIL for the whole of InferenceSession(): measured with
a 1 ms ticker thread that ran three times across twenty session creations,
and a CoreML build that logged no progress in ten minutes. In the server
process that would stop every thread - the request loop, the pings the comma
uses to tell a slow build from a dead link, the accept - for as long as CoreML
compiles. So the session lives here, in a child, and the server talks to it
over a pipe with the model's inputs and outputs in shared memory: the queues
gather straight into the shared block, the child runs, and the reply is a
few bytes. Tens of microseconds a frame on top of the model.

Spawned, never forked: a fork would carry the parent's GPU state into a
process that must not touch it. A spawn re-imports the parent's __main__, so
the parent has to be a module or a script file, which `-m jetlink.server.main`
and pytest are; a session fed on stdin cannot start a worker.
"""
from __future__ import annotations

import time
import traceback
from multiprocessing import shared_memory

import numpy as np

# One block, laid out as the child reports it: every input, then every output.
LAYOUT_ALIGN = 64


def layout(entries: list[tuple[str, tuple[int, ...], str]]) -> tuple[list[tuple[str, tuple[int, ...], str, int]], int]:
  """(name, shape, dtype, offset) per tensor, and the block size."""
  out = []
  offset = 0
  for name, shape, dtype in entries:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    out.append((name, tuple(shape), dtype, offset))
    offset += -(-nbytes // LAYOUT_ALIGN) * LAYOUT_ALIGN
  return out, max(offset, 1)


def views(block, laid: list[tuple[str, tuple[int, ...], str, int]]) -> dict[str, np.ndarray]:
  return {name: np.ndarray(shape, np.dtype(dtype), buffer=block.buf, offset=offset)
          for name, shape, dtype, offset in laid}


def main(conn, model: str, providers: list, log_severity: int) -> None:
  """Runs in the child. Protocol, parent's view:

      <- ('io', inputs, outputs)       once the session exists; each a list of (name, shape, dtype)
      -> ('attach', shm_name, laid_inputs, laid_outputs)
      <- ('ready', providers_in_use)
      -> ('run',)      <- ('ok', gpu_us) | ('error', text)
      -> ('close',)    child exits

  Any exception before 'ready' is sent as ('error', traceback) and the child
  exits; the parent turns it into the same exception the in-process path
  would have raised.
  """
  block = None
  try:
    import onnxruntime as ort

    from jetlink.server.backends.ort.engine import ORT_DTYPES

    so = ort.SessionOptions()
    so.log_severity_level = log_severity
    session = ort.InferenceSession(model, so, providers=providers)

    def describe(nodes):
      out = []
      for n in nodes:
        if n.type not in ORT_DTYPES:
          raise ValueError(f"{n.name} is {n.type}; add it to ORT_DTYPES")
        shape = tuple(int(d) for d in n.shape)
        if any(d <= 0 for d in shape):
          raise ValueError(f"{n.name} has a dynamic shape {n.shape}; jetlink builds fixed-shape engines")
        out.append((n.name, shape, np.dtype(ORT_DTYPES[n.type]).name))
      return out

    inputs = describe(session.get_inputs())
    outputs = describe(session.get_outputs())
    conn.send(('io', inputs, outputs))

    msg = conn.recv()
    if msg[0] != 'attach':
      raise RuntimeError(f"expected attach, got {msg[0]!r}")
    _, shm_name, laid_in, laid_out = msg
    block = shared_memory.SharedMemory(name=shm_name)
    feeds = views(block, laid_in)
    sinks = views(block, laid_out)
    names = [n for n, *_ in laid_out]
    conn.send(('ready', list(session.get_providers())))

    while True:
      msg = conn.recv()
      if msg[0] == 'close':
        break
      if msg[0] != 'run':
        conn.send(('error', f"unknown request {msg[0]!r}"))
        continue
      try:
        t0 = time.perf_counter()
        results = session.run(names, feeds)
        for name, value in zip(names, results, strict=True):
          np.copyto(sinks[name], np.asarray(value).reshape(sinks[name].shape), casting='unsafe')
        conn.send(('ok', int((time.perf_counter() - t0) * 1e6)))
      except Exception as e:
        conn.send(('error', f"{type(e).__name__}: {e}"))
  except Exception:
    try:
      conn.send(('error', traceback.format_exc()))
    except OSError:
      pass
  finally:
    if block is not None:
      block.close()
    conn.close()
