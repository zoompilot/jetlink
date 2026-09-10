"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

onnxruntime: CoreML on a Mac, CUDA or plain CPU anywhere else.

On Apple silicon the model runs in one CoreML session on the GPU
(`--device coreml`, the default there): 43 ms round trip at 20 Hz on an
M1 Pro, p99 44, parity gate passed. `--device ane` allows every unit, the
Neural Engine included, and carries the two rewrites that make the Neural
Engine correct (jetlink.onnx_patch, measured 2026-09-08): a Gather with a
negative index gathers garbage there, so `normalize_gather_indices` writes
every such index from the front; and its fp16 LayerNormalization overflows
on this model's residual stream, so `layernorm_in_fp32` runs the policy's
LayerNormalizations in fp32, which CoreML places off the Neural Engine. With
both, the gate passes with the GPU path's precision and the round trip is
33 ms back to back. It is not the default because at 20 Hz, a frame every
50 ms with the units idle in between, the same session measured 45 ms with
a p99 of 59 on the M1 Pro against the GPU's 43 and 44: every CoreML unit
pays a cost on the first request after an idle gap, and the Neural Engine
pays more of it. A faster Mac may not; measure it with
scripts/bench_link.py --rate 20 before choosing (docs/platforms.md).
Every session costs minutes of compile that onnxruntime's cache directory
did not shorten.

Elsewhere it is one session with the plain graph. The ONNX gets the same
surgery TensorRT's build does: the `org.tinygrad` layout op stripped,
because onnxruntime rejects a domain it does not know, and the uint8 images
retyped to fp16, which is what the queues already produce, so no per-frame
cast is needed. CUDA through onnxruntime is a fallback for a laptop without
TensorRT, not a peer of it; CPU is for a bench with a small model.

The session runs in a worker process (worker.py): onnxruntime holds the GIL
while it creates a session, and a ten-minute compile with the GIL held would
stop the server answering anything. The frame's inputs and outputs sit in
shared memory, so the cost is a message each way. The server process never
imports onnxruntime itself (see `quiet`).

The artifact is a directory: the prepared ONNX, a manifest naming the
session(s), and onnxruntime's cache. A load needs nothing else on disk.
"""
from __future__ import annotations

import json
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

MANIFEST = 'sessions.json'

PROVIDERS = {
  'coreml': 'CoreMLExecutionProvider',
  'ane': 'CoreMLExecutionProvider',
  'cuda': 'CUDAExecutionProvider',
  'cpu': 'CPUExecutionProvider',
}

# CoreML compute units per device name. `ane` is correct only with the two
# rewrites in the module docstring, which the build applies for it.
COREML_UNITS = {'coreml': 'CPUAndGPU', 'ane': 'ALL'}

# What a CoreML compile of the big model took on an M1 Pro. The progress
# fraction is elapsed time against this, honestly labelled, because CoreML
# reports nothing until it is done and the comma's bar has to move.
EXPECTED_COREML_SECONDS = 660.0


def quiet(ort) -> None:
  """Turn onnxruntime's telemetry off in a process that has imported it.

  The macOS wheel carries Microsoft's events SDK, which uploads over HTTP
  from a worker thread of its own. A response that lands as the process
  exits is dispatched through a mutex static destruction has already torn
  down, and the process aborts: `recursive_mutex lock failed` from
  `Microsoft::Applications::Events::HttpClientManager::onHttpResponse` in
  the crash report, one test run in three. Disabling the events at import
  was not enough, an event logged by the import itself still uploads, so
  the server process never imports onnxruntime at all: the version comes
  from the package metadata and the providers from a probe in a child
  (`available_providers`). The worker calls this, and a car has no business
  making the request in the first place.
  """
  disable = getattr(ort, 'disable_telemetry_events', None)
  if disable is not None:
    disable()


def _probe_providers(conn) -> None:
  # runs in a child: the one import of onnxruntime the server never makes
  try:
    import onnxruntime as ort
    quiet(ort)
    conn.send(list(ort.get_available_providers()))
  except Exception as e:
    conn.send(e)
  finally:
    conn.close()


def available_providers() -> list[str]:
  """onnxruntime's providers on this machine, asked in a spawned child."""
  import multiprocessing as mp
  ctx = mp.get_context('spawn')
  parent, child = ctx.Pipe()
  proc = ctx.Process(target=_probe_providers, args=(child,), name='jetlink-ort-probe', daemon=True)
  proc.start()
  child.close()
  try:
    if not parent.poll(60):
      raise RuntimeError('onnxruntime did not answer the provider probe in 60 s')
    answer = parent.recv()
  finally:
    proc.join(5)
    parent.close()
  if isinstance(answer, Exception):
    raise RuntimeError(f"onnxruntime could not be imported: {answer}") from answer
  return answer


def runtime_version() -> str:
  from importlib.metadata import version
  return version('onnxruntime')


def _pick_device(providers: list[str], device: str) -> str:
  have = set(providers)
  if device in ('auto', '', None):
    # never `ane` on auto: it is the measured opt-in, see the module docstring
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


def _cache_key(out_path: Path, part: str) -> str:
  # onnxruntime wants the key alphanumeric and under 64 characters
  return re.sub(r'[^A-Za-z0-9]', '', out_path.stem + part)[:63]


def _prepared_model(onnx_path: Path, for_ane: bool):
  """The ONNX as onnxruntime will see it, in memory."""
  import onnx

  from jetlink.onnx_patch import (
    layernorm_in_fp32,
    needs_patch,
    normalize_gather_indices,
    patch_uint8_inputs,
    strip_tinygrad_ops,
    vision_nodes,
  )
  model = onnx.load(str(onnx_path))
  stripped = strip_tinygrad_ops(model)
  patched = needs_patch(model)
  if patched:
    patch_uint8_inputs(model)
  gathers = normalize_gather_indices(model)
  norms = 0
  if for_ane:
    policy = {n.name for n in model.graph.node} - vision_nodes(model)
    norms = layernorm_in_fp32(model, only=policy)
  log.info("prepared %s: stripped %d tinygrad op(s), %s, %d negative Gather index(es) normalized, "
           "%d LayerNormalization(s) in fp32", onnx_path.name, stripped,
           'images retyped to fp16' if patched else 'inputs left as declared', gathers, norms)
  return model


def _with_cache_key(model, key: str):
  # The compiled-model cache is looked up by this rather than by a hash of the
  # file, so the same key finds the same compile after a move.
  entry = next((p for p in model.metadata_props if p.key == 'CACHE_KEY'), None) or model.metadata_props.add()
  entry.key, entry.value = 'CACHE_KEY', key
  return model


def load_ticker(report, message):
  """One tick of a load: the progress a client draws, and a line for the log.

  Progress goes out every tick, because the comma and the app would otherwise
  sit on stage load, frac 0, "deserializing engine" for the nine minutes
  CoreML takes, which reads as a hung server. The log does not: a tick every
  5 s is over a hundred identical lines per load and the Logs view has nothing
  else in it, so it is info on the first tick and once a minute after that,
  debug for the rest.
  """
  said = [-1]

  def tick(elapsed: float) -> None:
    minute = int(elapsed // 60)
    log.log(logging.INFO if minute != said[0] else logging.DEBUG,
            "still creating the onnxruntime sessions, %.0f s", elapsed)
    said[0] = minute
    if report is not None:
      report('load', 0.0, message(elapsed))

  return tick


class OrtBackend:
  name = 'ort'
  suffix = '.ortcache'

  def __init__(self, device: str = 'auto', providers: list[str] | None = None):
    self._version = runtime_version()
    self.device = _pick_device(available_providers() if providers is None else providers, device)
    if self.device == 'cpu':
      log.warning("onnxruntime on the CPU will not make the frame budget; fine for a bench, not a car")

  @property
  def runtime_version(self) -> str:
    return self._version

  def device_tag(self) -> str:
    return sanitize(f"{self.device}-{gpu_name()}")

  def tag(self) -> str:
    return f"ort{sanitize(self.runtime_version)}.{self.device_tag()}"

  def describe(self) -> dict:
    return {'backend': self.name, 'runtime_version': self.runtime_version, 'device': self.device_tag()}

  # -- sessions ---------------------------------------------------------------

  @property
  def _on_coreml(self) -> bool:
    return self.device in COREML_UNITS

  def _providers(self, units: str | None, compiled_dir: Path | None) -> list:
    if self._on_coreml:
      opts = {'ModelFormat': 'MLProgram', 'MLComputeUnits': units or COREML_UNITS[self.device]}
      if compiled_dir is not None:
        opts['ModelCacheDirectory'] = str(compiled_dir)
      return [(PROVIDERS['coreml'], opts), PROVIDERS['cpu']]
    if self.device == 'cuda':
      return [(PROVIDERS['cuda'], {'device_id': 0}), PROVIDERS['cpu']]
    return [PROVIDERS['cpu']]

  def _plan(self, artifact: Path, manifest: list[dict]) -> list[tuple[Path, list]]:
    """[(model path, providers)] for the worker, from a manifest entry per session."""
    plan = []
    for entry in manifest:
      compiled = artifact / entry['cache'] if entry.get('cache') else None
      plan.append((artifact / entry['model'], self._providers(entry.get('units'), compiled)))
    return plan

  def _engine(self, artifact: Path, manifest: list[dict], on_tick=None):
    from jetlink.server.backends.ort.engine import OrtEngine
    # log severity 3: errors only; the CoreML partitioner is chatty
    return OrtEngine(self._plan(artifact, manifest), self.device, log_severity=3, on_tick=on_tick)

  # -- build ------------------------------------------------------------------

  def _stage(self, onnx_path: Path, staged: Path, out_path: Path) -> list[dict]:
    """Write the prepared model into `staged` and return the manifest.

    The manifest is a list because the worker runs sessions as a chain and
    a split graph was measured through it; one session is what ships.
    """
    import onnx

    coreml = self._on_coreml
    model = _prepared_model(onnx_path, for_ane=self.device == 'ane')
    onnx.save(_with_cache_key(model, _cache_key(out_path, 'model')), str(staged / 'model.onnx'))
    manifest = [{'model': 'model.onnx', 'units': COREML_UNITS[self.device] if coreml else None,
                 'cache': 'coreml' if coreml else None}]
    for entry in manifest:
      if entry['cache']:
        (staged / entry['cache']).mkdir()
    (staged / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest

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
      manifest = self._stage(onnx_path, staged, out_path)
      report('patch', 1.0, 'prepared')

      report('build', 0.0, self._build_message(0.0))
      engine = self._engine(staged, manifest, on_tick=lambda elapsed: report(
        'build', self._build_fraction(elapsed), self._build_message(elapsed)))
      try:
        report('build', 1.0, f'sessions created in {time.time() - t0:.0f} s')
        # Prove it runs before calling it built; a partition that fell back to
        # the CPU in full would still "work", so the log line says what it used.
        engine.run()
        log.info("onnxruntime providers in use: %s", engine.providers)
        providers = engine.providers
      finally:
        engine.close()

      if out_path.exists():
        shutil.rmtree(out_path)
      shutil.move(str(staged), str(out_path))

    meta = {
      'backend': self.name,
      'onnxruntime': self.runtime_version,
      'device': self.device_tag(),
      'sessions': manifest,
      'providers': providers,
      'build_seconds': round(time.time() - t0, 1),
      'onnx': onnx_path.name,
      'built_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      **(meta_extra or {}),
    }
    write_sidecar(out_path, meta)
    report('build', 1.0, f"done in {meta['build_seconds']}s")
    return out_path

  def _load_message(self, elapsed: float) -> str:
    if self._on_coreml:
      return (f'creating the CoreML session, {elapsed / 60:.0f} min elapsed; the big model takes '
              f'about {EXPECTED_COREML_SECONDS / 60:.0f} min on an M1 Pro')
    return f'creating the onnxruntime session ({self.device})'

  def _build_message(self, elapsed: float) -> str:
    if self._on_coreml:
      return (f'compiling for CoreML, {elapsed / 60:.0f} min elapsed; the big model takes '
              f'{EXPECTED_COREML_SECONDS / 60:.0f} min on an M1 Pro')
    return f'creating the onnxruntime session ({self.device})'

  def _build_fraction(self, elapsed: float) -> float:
    return min(0.95, elapsed / EXPECTED_COREML_SECONDS) if self._on_coreml else 0.5

  # -- load -------------------------------------------------------------------

  def load(self, artifact: Path, report: ProgressFn | None = None):
    artifact = Path(artifact)
    try:
      manifest = json.loads((artifact / MANIFEST).read_text())
    except (OSError, ValueError) as e:
      raise ArtifactInvalid(f"{artifact}: no readable {MANIFEST} inside ({e})") from e
    if not isinstance(manifest, list) or not manifest:
      raise ArtifactInvalid(f"{artifact}: {MANIFEST} names no sessions")
    for entry in manifest:
      if not (artifact / entry['model']).is_file():
        raise ArtifactInvalid(f"{artifact}: no {entry['model']} inside")
      cache = artifact / entry['cache'] if entry.get('cache') else None
      # An empty cache would make onnxruntime recompile for minutes under a
      # "loading engine" that never moves. Rebuild instead, which reports
      # progress and ends with a cache.
      if cache is not None and (not cache.is_dir() or not any(cache.iterdir())):
        raise ArtifactInvalid(f"{artifact}: the CoreML cache for {entry['model']} is empty")
    t0 = time.time()
    if self._on_coreml:
      log.info("creating the CoreML session; measured at %.0f min on an M1 Pro, cache or no cache",
               EXPECTED_COREML_SECONDS / 60)
    engine = self._engine(artifact, manifest, on_tick=load_ticker(report, self._load_message))
    log.info("onnxruntime sessions on %s in %.1f s, providers %s", self.device, time.time() - t0,
             engine.providers)
    return engine
