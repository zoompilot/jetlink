"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

onnxruntime: CoreML on a Mac, CUDA or plain CPU anywhere else.

Measured on an M1 Pro the CoreML provider runs the 766 MB model in 39 ms a
frame on the GPU, against 65 ms for tinygrad on the same Metal GPU, both
correct. The price is the session: CoreML compiles the model for ten minutes
every time a process creates one, and onnxruntime's ModelCacheDirectory did
not shorten the second session in the measurement (docs/platforms.md has the
numbers). So this is the faster frame and the slower start, and tinygrad is
the Mac's default; pick this with --backend ort for the frame time once the
server is long-lived enough to amortise the start.

The artifact is a directory holding the patched ONNX and onnxruntime's cache,
so a load needs nothing else on disk and a future onnxruntime that does reuse
the cache gets the benefit without a rebuild.

The session runs in a worker process (worker.py): onnxruntime holds the GIL
while it creates a session, and a ten-minute compile with the GIL held would
stop the server answering anything. The frame's inputs and outputs sit in
shared memory, so the cost is a message each way.

The ONNX gets the same surgery TensorRT's build does (jetlink.onnx_patch): the
`org.tinygrad` layout op stripped, because onnxruntime rejects a domain it
does not know, and the uint8 images retyped to fp16, which is what the queues
already produce, so no per-frame cast is needed.

CUDA through onnxruntime is a fallback for a laptop without TensorRT, not a
peer of it; CPU is for a bench with a small model.
"""
from __future__ import annotations

import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from jetlink.server.backends.base import (
  ArtifactInvalid,
  ProgressFn,
  sanitize,
  write_sidecar,
)
from jetlink.server.platform import gpu_name

log = logging.getLogger('jetlink.ort')

MODEL = 'model.onnx'
COMPILED = 'coreml'

PROVIDERS = {
  'coreml': 'CoreMLExecutionProvider',
  'cuda': 'CUDAExecutionProvider',
  'cpu': 'CPUExecutionProvider',
}

# What a CoreML compile of the big model took on an M1 Pro. The progress
# fraction is elapsed time against this, honestly labelled, because CoreML
# reports nothing until it is done and the comma's bar has to move.
EXPECTED_COREML_SECONDS = 600.0


def _pick_device(ort, device: str) -> str:
  have = set(ort.get_available_providers())
  if device in ('auto', '', None):
    order = ('coreml', 'cuda', 'cpu') if sys.platform == 'darwin' else ('cuda', 'cpu')
    for d in order:
      if PROVIDERS[d] in have:
        return d
    raise RuntimeError(f"onnxruntime has none of {list(PROVIDERS.values())}; has {sorted(have)}")
  d = device.lower()
  if d not in PROVIDERS:
    raise ValueError(f"onnxruntime device must be one of {list(PROVIDERS)}, not {device!r}")
  if PROVIDERS[d] not in have:
    raise RuntimeError(f"onnxruntime here has no {PROVIDERS[d]} (has {sorted(have)})")
  return d


def _cache_key(out_path: Path) -> str:
  # onnxruntime wants the key alphanumeric and under 64 characters
  return re.sub(r'[^A-Za-z0-9]', '', out_path.stem)[:63]


def _prepared_model(onnx_path: Path, dest: Path, cache_key: str) -> None:
  """The ONNX as onnxruntime will see it, written to `dest`."""
  import onnx

  from jetlink.onnx_patch import needs_patch, patch_uint8_inputs, strip_tinygrad_ops
  model = onnx.load(str(onnx_path))
  stripped = strip_tinygrad_ops(model)
  patched = needs_patch(model)
  if patched:
    patch_uint8_inputs(model)
  # The compiled-model cache is looked up by this rather than by a hash of the
  # file, so the same key finds the same compile after a move.
  entry = next((p for p in model.metadata_props if p.key == 'CACHE_KEY'), None) or model.metadata_props.add()
  entry.key, entry.value = 'CACHE_KEY', cache_key
  onnx.save(model, str(dest))
  log.info("prepared %s: stripped %d tinygrad op(s), %s", onnx_path.name, stripped,
           'images retyped to fp16' if patched else 'inputs left as declared')


class OrtBackend:
  name = 'ort'
  suffix = '.ortcache'

  def __init__(self, device: str = 'auto'):
    import onnxruntime as ort
    self.ort = ort
    self.device = _pick_device(ort, device)
    if self.device == 'cpu':
      log.warning("onnxruntime on the CPU will not make the frame budget; fine for a bench, not a car")

  @property
  def runtime_version(self) -> str:
    return self.ort.__version__

  def device_tag(self) -> str:
    return sanitize(f"{self.device}-{gpu_name()}")

  def tag(self) -> str:
    return f"ort{sanitize(self.runtime_version)}.{self.device_tag()}"

  def describe(self) -> dict:
    return {'backend': self.name, 'runtime_version': self.runtime_version, 'device': self.device_tag()}

  def _providers(self, compiled_dir: Path | None) -> list:
    if self.device == 'coreml':
      # CPUAndGPU, not ALL: with the Neural Engine allowed, the big model ran
      # in 25 ms and was wrong (whole-output correlation 0.91 to 0.97 against
      # the CPU provider, CoreML logging "ANE model load has failed"); on the
      # GPU alone it runs in 39 ms and matches to 0.999998. Measured 2026-09-07
      # on an M1 Pro, docs/platforms.md.
      opts = {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}
      if compiled_dir is not None:
        opts['ModelCacheDirectory'] = str(compiled_dir)
      return [(PROVIDERS['coreml'], opts), PROVIDERS['cpu']]
    if self.device == 'cuda':
      return [(PROVIDERS['cuda'], {'device_id': 0}), PROVIDERS['cpu']]
    return [PROVIDERS['cpu']]

  def _engine(self, model: Path, compiled_dir: Path | None, on_tick=None):
    from jetlink.server.backends.ort.engine import OrtEngine
    # log severity 3: errors only; the CoreML partitioner is chatty
    return OrtEngine(str(model), self._providers(compiled_dir), self.device, log_severity=3, on_tick=on_tick)

  def build(self, onnx_path: Path, out_path: Path, report: ProgressFn | None = None,
            meta_extra: dict | None = None) -> Path:
    onnx_path, out_path = Path(onnx_path), Path(out_path)
    report = report or (lambda *_: None)
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp:
      staged = Path(tmp) / 'artifact'
      staged.mkdir()
      report('patch', 0.0, 'preparing the model for onnxruntime')
      _prepared_model(onnx_path, staged / MODEL, _cache_key(out_path))
      report('patch', 1.0, 'prepared')

      compiled = staged / COMPILED if self.device == 'coreml' else None
      if compiled is not None:
        compiled.mkdir()
      report('build', 0.0, self._build_message(0.0))
      engine = self._engine(staged / MODEL, compiled, on_tick=lambda elapsed: report(
        'build', self._build_fraction(elapsed), self._build_message(elapsed)))
      try:
        report('build', 1.0, f'session created in {time.time() - t0:.0f} s')
        # Prove it runs before calling it built; a partition that fell back to
        # the CPU in full would still "work", so the log line says what it used.
        engine.run()
        log.info("onnxruntime providers in use: %s", engine.providers)
      finally:
        engine.close()

      if out_path.exists():
        shutil.rmtree(out_path)
      shutil.move(str(staged), str(out_path))

    meta = {
      'backend': self.name,
      'onnxruntime': self.runtime_version,
      'device': self.device_tag(),
      'providers': [p if isinstance(p, str) else p[0] for p in self._providers(None)],
      'build_seconds': round(time.time() - t0, 1),
      'onnx': onnx_path.name,
      'built_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      **(meta_extra or {}),
    }
    write_sidecar(out_path, meta)
    report('build', 1.0, f"done in {meta['build_seconds']}s")
    return out_path

  def _build_message(self, elapsed: float) -> str:
    if self.device == 'coreml':
      return (f'compiling for CoreML, {elapsed / 60:.0f} min elapsed; the big model takes '
              f'{EXPECTED_COREML_SECONDS / 60:.0f} min on an M1 Pro')
    return f'creating the onnxruntime session ({self.device})'

  def _build_fraction(self, elapsed: float) -> float:
    return min(0.95, elapsed / EXPECTED_COREML_SECONDS) if self.device == 'coreml' else 0.5

  def load(self, artifact: Path):
    artifact = Path(artifact)
    model = artifact / MODEL
    if not model.is_file():
      raise ArtifactInvalid(f"{artifact}: no {MODEL} inside")
    compiled = None
    if self.device == 'coreml':
      compiled = artifact / COMPILED
      # An empty cache would make onnxruntime recompile for twenty minutes
      # under a "loading engine" that never moves. Rebuild instead, which
      # reports progress and ends with a cache.
      if not compiled.is_dir() or not any(compiled.iterdir()):
        raise ArtifactInvalid(f"{artifact}: the CoreML cache is empty")
    t0 = time.time()
    if self.device == 'coreml':
      log.info("creating the CoreML session; measured at %.0f min on an M1 Pro, cache or no cache",
               EXPECTED_COREML_SECONDS / 60)
    engine = self._engine(model, compiled, on_tick=lambda elapsed: log.info(
      "still creating the onnxruntime session, %.0f s", elapsed))
    log.info("onnxruntime session on %s in %.1f s, providers %s", self.device, time.time() - t0,
             engine.providers)
    return engine
