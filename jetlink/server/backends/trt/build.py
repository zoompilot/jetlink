"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

ONNX -> TensorRT engine, with progress.

A plan is specific to the TensorRT version, the GPU architecture and the build
flags, so the cache key encodes all three: a new JetPack or a different Jetson
rebuilds instead of loading something that cannot run.

The build takes ~160 s for the big model, so progress is streamed back to the
comma, where it shows up the way a model download does.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt

log = logging.getLogger('jetlink.builder')

ProgressFn = Callable[[str, float, str], None]

DEFAULT_CACHE = Path(os.environ.get('JETLINK_CACHE', '/mnt/data/jetlink'))
# A ceiling, not an allocation: TensorRT picks tactics that fit inside it. A
# flat 4 GB on an 8 GB Orin met the OOM killer on a 1.76 GB model, so size it
# from what is free.
MAX_WORKSPACE_BYTES = 4 << 30
MIN_WORKSPACE_BYTES = 256 << 20
WORKSPACE_FRACTION = 0.4
# One per registry entry: a rebuild costs minutes and a plan under 2 GB of a
# 900 GB disk, so a smaller cap makes an A/B rebuild on every switch.
KEEP_PLANS = 6


def available_bytes() -> int:
  """Memory a TensorRT workspace can live in.

  MemAvailable, so free plus what the kernel would reclaim. Swap does not count:
  on Tegra the GPU's allocations are pinned system RAM and cannot page out, and
  this box has 25 GB of swap to be fooled by.
  """
  try:
    with open('/proc/meminfo') as f:
      for line in f:
        key, _, rest = line.partition(':')
        if key == 'MemAvailable':
          return int(rest.split()[0]) * 1024
  except OSError:
    pass
  return 0


def workspace_bytes() -> int:
  free = available_bytes()
  if free <= 0:
    return MAX_WORKSPACE_BYTES
  return max(MIN_WORKSPACE_BYTES, min(MAX_WORKSPACE_BYTES, int(free * WORKSPACE_FRACTION)))


def _sanitize(s: str) -> str:
  return re.sub(r'[^A-Za-z0-9._-]', '_', s)


def device_tag() -> str:
  """Identifies the hardware a plan is valid for.

  The compute capability is the part that matters; the name makes the cache
  filename readable. From CUDA, not /proc/device-tree, which the container has
  no mount for.
  """
  try:
    from jetlink.server import cudart
    name, cc_major, cc_minor = cudart.device_name()
    return _sanitize(f"{name}-sm{cc_major}{cc_minor}")
  except Exception:
    return 'unknown'


@dataclass
class CacheEntry:
  plan_path: Path
  meta_path: Path

  @property
  def exists(self) -> bool:
    return self.plan_path.is_file() and self.meta_path.is_file()

  def meta(self) -> dict:
    return json.loads(self.meta_path.read_text())

  def write_meta(self, meta: dict) -> None:
    self.meta_path.write_text(json.dumps(meta, indent=2))


# What the server loaded last, so a fresh process can preload it. Beside the
# caches rather than in them: it describes the server, not a plan.
LAST_LOADED = 'last-loaded.json'

# Kernel timings mostly do not depend on the model: a warm cache cut a Lebowski
# build from 254 s to 173 s. Keyed like the plans, because a timing from another
# version or chip is not one. Advisory, so every path here fails open.
TIMING_CACHE = 'timing'


class EngineCache:
  def __init__(self, root: Path = DEFAULT_CACHE):
    self.root = Path(root)
    self.engines = self.root / 'engines'
    self.models = self.root / 'models'
    for d in (self.engines, self.models):
      d.mkdir(parents=True, exist_ok=True)

  def key(self, model_sha256: str) -> str:
    self._validate_sha256(model_sha256)
    return f"{model_sha256[:16]}.trt{_sanitize(trt.__version__)}.{device_tag()}"

  def entry(self, model_sha256: str) -> CacheEntry:
    k = self.key(model_sha256)
    return CacheEntry(self.engines / f"{k}.plan", self.engines / f"{k}.json")

  def model_path(self, model_sha256: str) -> Path:
    self._validate_sha256(model_sha256)
    return self.models / f"{model_sha256[:16]}.onnx"

  @staticmethod
  def _validate_sha256(value: str) -> None:
    # Model identities arrive from the peer and become filesystem paths.
    if re.fullmatch(r'[0-9a-f]{64}', value) is None:
      raise ValueError('model identity must be a lowercase SHA-256 digest')

  def inventory(self) -> list[str]:
    """Model identities with plans compatible with this GPU and TensorRT."""
    found = []
    for meta in self.engines.glob('*.json'):
      try:
        sha = json.loads(meta.read_text())['spec']['sha256']
        if (isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha)
            and self.entry(sha).meta_path == meta and self.entry(sha).exists):
          found.append(sha)
      except (OSError, ValueError, KeyError, TypeError):
        continue
    return sorted(set(found))

  def remember_loaded(self, sha256: str, frame_skip: int) -> None:
    """Record what is loaded, for the next process to preload.

    frame_skip goes with it: the spec served is stamped with it, so preloading
    under another value hands the next client a spec it did not ask for.
    """
    try:
      (self.root / LAST_LOADED).write_text(json.dumps({'sha256': sha256, 'frame_skip': frame_skip}))
    except OSError:
      pass   # a read-only cache still serves; it just cannot preload next time

  def last_loaded(self) -> tuple[str, int] | None:
    try:
      d = json.loads((self.root / LAST_LOADED).read_text())
      sha, skip = d['sha256'], int(d['frame_skip'])
    except (OSError, ValueError, KeyError, TypeError):
      return None
    return (sha, skip) if re.fullmatch(r'[0-9a-f]{64}', sha) else None

  def timing_cache(self) -> Path:
    return self.engines / f"{TIMING_CACHE}.trt{_sanitize(trt.__version__)}.{device_tag()}.cache"

  def prune(self, keep: int = KEEP_PLANS, protect: Path | None = None) -> None:
    """Keep the newest few plans; each is ~770 MB.

    `protect` is never pruned whatever its mtime says: this box boots at 1970
    without NTP, so a plan built offroad looks older than everything on disk and
    a fresh build would be the first one deleted.
    """
    plans = [p for p in self.engines.glob('*.plan') if protect is None or p != protect]
    plans.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in plans[max(keep - (protect is not None), 0):]:
      p.unlink(missing_ok=True)
      p.with_suffix('.json').unlink(missing_ok=True)

  def sweep_temp(self, max_age: float = 6 * 3600) -> None:
    """Drop build directories a crashed or killed build left behind.

    build_engine stages the plan in a TemporaryDirectory inside engines/, which
    prune() does not glob.
    """
    now = time.time()
    for d in self.engines.glob('tmp*'):
      if not d.is_dir():
        continue
      try:
        if now - d.stat().st_mtime < max_age:
          continue
        shutil.rmtree(d, ignore_errors=True)
      except OSError:
        pass


class _Monitor(trt.IProgressMonitor):
  """Turns TensorRT's build phases into a single 0..1 fraction."""

  def __init__(self, report: ProgressFn):
    super().__init__()
    self.report = report
    self.phases: dict[str, tuple[int, int]] = {}
    self.root: str | None = None

  def _emit(self) -> None:
    if self.root and self.root in self.phases:
      step, total = self.phases[self.root]
      frac = (step / total) if total else 0.0
      self.report('build', min(max(frac, 0.0), 1.0), self.root)

  def phase_start(self, phase_name, parent_phase, num_steps):
    if parent_phase is None:
      self.root = phase_name
    self.phases[phase_name] = (0, num_steps)
    self._emit()

  def step_complete(self, phase_name, step):
    total = self.phases.get(phase_name, (0, 0))[1]
    self.phases[phase_name] = (step, total)
    self._emit()
    return True  # False would abort the build

  def phase_finish(self, phase_name):
    self.phases.pop(phase_name, None)
    if phase_name == self.root:
      self.root = None


def _load_timing_cache(config, path: str | Path | None):
  """Seed the builder's tactic timings from a previous build.

  Fails open: another TensorRT version's cache is rejected and a killed build's
  is truncated, and either way the build runs, just cold.
  """
  if path is None:
    return None
  blob = b''
  try:
    blob = Path(path).read_bytes()
  except OSError:
    pass
  try:
    cache = config.create_timing_cache(blob)
    if cache is not None:
      config.set_timing_cache(cache, ignore_mismatch=False)
    return cache
  except Exception as e:
    log.warning("timing cache unusable (%s), building cold", e)
    return None


def _save_timing_cache(cache, path: str | Path | None) -> None:
  if cache is None or path is None:
    return
  try:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic like the plan: a build killed mid-write would leave a truncated
    # cache for the next one to read.
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_bytes(memoryview(cache.serialize()))
    tmp.replace(p)
  except (OSError, AttributeError) as e:
    log.warning("could not write the timing cache: %s", e)


def build_engine(onnx_path: str | Path, out_path: str | Path,
                 report: ProgressFn | None = None,
                 fp16: bool = True, optimization_level: int = 3,
                 workspace: int | None = None,
                 meta_extra: dict | None = None,
                 timing_cache: str | Path | None = None) -> Path:
  """Patch, parse and build. Writes the plan atomically.

  `meta_extra` lands in the sidecar json next to the plan; the server keeps
  the model spec there so a later load needs neither the ONNX nor a parser.
  """
  onnx_path, out_path = Path(onnx_path), Path(out_path)
  report = report or (lambda *_: None)
  workspace = workspace_bytes() if workspace is None else workspace
  t0 = time.time()
  log.info("building with a %d MB workspace (%d MB available)",
           workspace >> 20, available_bytes() >> 20)

  logger = trt.Logger(trt.Logger.WARNING)
  trt.init_libnvinfer_plugins(logger, '')

  # Not at module scope: pulls in the onnx package, which a comma running the
  # tests does not have.
  from jetlink.onnx_patch import patch_file

  with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp:
    report('patch', 0.0, 'retyping uint8 image inputs to fp16')
    patched = Path(tmp) / 'patched.onnx'
    patch_file(str(onnx_path), str(patched))
    report('patch', 1.0, 'patched')

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    report('parse', 0.0, 'parsing onnx')
    if not parser.parse_from_file(str(patched)):
      errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
      raise RuntimeError("onnx parse failed:\n" + "\n".join(errs))
    report('parse', 1.0, f'{network.num_layers} layers')

    config = builder.create_builder_config()
    if fp16:
      config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.progress_monitor = _Monitor(report)
    cache = _load_timing_cache(config, timing_cache)

    report('build', 0.0, 'building engine')
    plan = builder.build_serialized_network(network, config)
    if plan is None:
      raise RuntimeError("TensorRT returned no engine; see the build log")
    _save_timing_cache(cache, timing_cache)

    staged = Path(tmp) / 'engine.plan'
    with open(staged, 'wb') as f:
      f.write(plan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(out_path))

  meta = {
    'trt_version': trt.__version__,
    'device': device_tag(),
    'fp16': fp16,
    'optimization_level': optimization_level,
    'build_seconds': round(time.time() - t0, 1),
    'onnx': onnx_path.name,
    'built_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    **(meta_extra or {}),
  }
  out_path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
  report('build', 1.0, f"done in {meta['build_seconds']}s")
  return out_path
