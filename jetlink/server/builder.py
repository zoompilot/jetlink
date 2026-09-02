"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

ONNX -> TensorRT engine, with progress.

A serialized TensorRT plan is not portable: it is specific to the TensorRT
version, the GPU architecture and the build flags. The cache key encodes all
three, so moving to a new JetPack or a different Jetson rebuilds rather than
silently loading something that will not run.

The build takes ~160 s for the big model, which is why progress is streamed
back to the comma rather than reported at the end - it shows up there the same
way a model download and compile does.
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
# Ceiling, not an allocation: TensorRT picks tactics that fit inside it. On an
# 8 GB Orin the old flat 4 GB let it choose tactics that, on top of the parsed
# weights, walked the builder into the OOM killer on a 1.76 GB model. Size it
# from what the machine actually has free instead.
MAX_WORKSPACE_BYTES = 4 << 30
MIN_WORKSPACE_BYTES = 256 << 20
WORKSPACE_FRACTION = 0.4


def available_bytes() -> int:
  """Memory a TensorRT workspace can actually live in.

  MemAvailable is the honest number - free plus what the kernel would reclaim.
  Swap deliberately does not count: on Tegra the GPU shares system RAM and its
  allocations are pinned, so they cannot page out. Counting the 25 GB of swap
  this Jetson happens to have would hand back the old flat 4 GB every time and
  walk the builder into the same OOM the sizing exists to avoid.
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

  A serialized plan is tied to the GPU architecture it was built for, so the
  compute capability is the part that actually matters; the device name makes
  the cache filename readable. Taken from CUDA rather than /proc/device-tree,
  which is not mounted inside the container.
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


class EngineCache:
  def __init__(self, root: Path = DEFAULT_CACHE):
    self.root = Path(root)
    self.engines = self.root / 'engines'
    self.models = self.root / 'models'
    for d in (self.engines, self.models):
      d.mkdir(parents=True, exist_ok=True)

  def key(self, model_sha256: str) -> str:
    return f"{model_sha256[:16]}.trt{_sanitize(trt.__version__)}.{device_tag()}"

  def entry(self, model_sha256: str) -> CacheEntry:
    k = self.key(model_sha256)
    return CacheEntry(self.engines / f"{k}.plan", self.engines / f"{k}.json")

  def model_path(self, model_sha256: str) -> Path:
    return self.models / f"{model_sha256[:16]}.onnx"

  def prune(self, keep: int = 2) -> None:
    """Keep only the newest few plans; each is ~770 MB."""
    plans = sorted(self.engines.glob('*.plan'), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in plans[keep:]:
      p.unlink(missing_ok=True)
      p.with_suffix('.json').unlink(missing_ok=True)


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


def build_engine(onnx_path: str | Path, out_path: str | Path,
                 report: ProgressFn | None = None,
                 fp16: bool = True, optimization_level: int = 3,
                 workspace: int | None = None,
                 meta_extra: dict | None = None) -> Path:
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

  # Imported here, not at module scope: it pulls in the onnx package, which the
  # server needs to build and a comma running the tests does not have.
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

    report('build', 0.0, 'building engine')
    plan = builder.build_serialized_network(network, config)
    if plan is None:
      raise RuntimeError("TensorRT returned no engine; see the build log")

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
