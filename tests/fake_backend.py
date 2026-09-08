"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A backend with no runtime behind it, for the session and cache tests.

The engine returns something deterministic that depends on the inputs, so a
wiring mistake between the queues and the engine cannot pass unnoticed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jetlink.server.backends.base import IO, ArtifactInvalid
from jetlink.spec import ModelSpec


class FakeEngine:
  def __init__(self, spec: ModelSpec):
    self.spec = spec
    self.inputs = {k: IO(k, tuple(v), np.dtype(np.float16)) for k, v in spec.input_shapes.items()}
    self.outputs = {'outputs': IO('outputs', (spec.output_nelem,), np.dtype(np.float16))}
    self._host = {k: np.zeros(v, np.float16) for k, v in spec.input_shapes.items()}
    self._out = np.zeros(spec.output_nelem, np.float16)
    self.last_gpu_us = 1234
    self.calls = 0
    self.warmed = 0
    self.closed = False
    self.nonfinite = False

  def host_input(self, name):
    return self._host[name]

  def run(self):
    self.calls += 1
    # Fold a couple of inputs into the output so the test can verify the queues
    # actually fed the engine. Sample rather than reduce: a float16 sum over
    # 393216 elements overflows to inf, which the server correctly rejects.
    self._out[:] = self._host['img'][0, 0, 0, 0]
    self._out[0] = self._host['features_buffer'][0, 0, 0, 0]
    self._out[1] = np.float16(self.calls)
    self._out[2] = self._host['img'][0, 6, 0, 0]
    if self.nonfinite:
      self._out[5] = np.float16('nan')
    return {'outputs': self._out}

  def warm(self):
    self.warmed += 1
    self.run()
    return 'fake engine warmed'

  def close(self):
    self.closed = True


class FakeBackend:
  """Builds an artifact that is a json file naming the model, and loads a
  FakeEngine shaped by the spec it was given. `invalid_loads` makes the next
  loads raise ArtifactInvalid, which is how the self-healing path is tested."""
  name = 'fake'
  suffix = '.fake'

  def __init__(self, spec: ModelSpec | None = None, version: str = '0.1', device: str = 'test'):
    self.spec = spec
    self.version = version
    self.device = device
    self.builds: list[Path] = []
    self.loads: list[Path] = []
    self.invalid_loads = 0
    self.engines: list[FakeEngine] = []

  def tag(self) -> str:
    return f"fake{self.version}.{self.device}"

  def describe(self) -> dict:
    return {'backend': self.name, 'runtime_version': self.version, 'device': self.device}

  def build(self, onnx_path: Path, out_path: Path, report=None, meta_extra=None) -> Path:
    report = report or (lambda *_: None)
    report('build', 0.0, 'faking a build')
    out_path = Path(out_path)
    out_path.write_text(json.dumps({'onnx': str(onnx_path), 'tag': self.tag()}))
    out_path.with_suffix('.json').write_text(json.dumps({'backend': self.name, **(meta_extra or {})}))
    self.builds.append(out_path)
    report('build', 1.0, 'done')
    return out_path

  def load(self, artifact: Path) -> FakeEngine:
    self.loads.append(Path(artifact))
    if self.invalid_loads > 0:
      self.invalid_loads -= 1
      raise ArtifactInvalid(f"{artifact}: made invalid by the test")
    if not Path(artifact).is_file():
      raise FileNotFoundError(artifact)
    if self.spec is None:
      raise RuntimeError('FakeBackend needs a spec to load an engine')
    engine = FakeEngine(self.spec)
    self.engines.append(engine)
    return engine
